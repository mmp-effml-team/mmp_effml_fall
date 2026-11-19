# Домашнее задание. FSDP\* своими руками

Дата выдачи: **17 ноября**. Мягкий дедлайн: **9 декабря 15:00**. Жёсткий дедлайн: **16 декабря 15:00**.
Стоимость: **10 баллов**. После мягкого дедлайна штраф 1 балл за каждый день просрочки, после жёсткого решения
не принимаются.

FSDP (ZeRO-3) хранит на каждом из N рангов только 1/N весов, градиентов и состояний оптимизатора. Полные веса
блока собираются all-gather'ом прямо перед его forward и backward и сразу освобождаются, а градиенты сводятся
reduce-scatter'ом: каждый ранг получает усреднённый градиент своего шарда. В этом задании вы напишете такой FSDP
сами внутри учебного фреймворка `mtf` и сравните его с FSDP2 из PyTorch (`torch.distributed.fsdp.fully_shard`),
который в `mtf` уже есть.

Задание повторяет домашку курса Efficient Deep Learning Systems (ШАД), неделя 5:
https://github.com/mryab/efficient-dl-systems/tree/main/week05_fsdp/homework. Скелет тот же: параметр с шардом
и полной копией, хуки вокруг forward и backward, prefetch. Но всё работает на CPU с бэкендом gloo, 4 и 8
процессов, GPU не нужен. Вместо CUDA streams для overlap — асинхронные коллективы gloo.

Корректность проверяется эквивалентностью: обучение с вашим FSDP должно повторять обучение в одном процессе и
совпадать с FSDP2 — тот же loss и та же норма градиента на каждом шаге.

\*FSDP (здесь и далее) продукт Meta — признана экстремистской организацией и запрещена на территории Российской Федерации.

Почитать:
- PyTorch FSDP: https://arxiv.org/abs/2304.11277; ZeRO: https://arxiv.org/abs/1910.02054
- FSDP2: RFC https://github.com/pytorch/pytorch/issues/114299 и документация
  https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- Ultra-Scale Playbook, раздел про ZeRO: https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=zero_redundancy_optimizer_(zero)
- Семинар 10: хуки FSDP2, streams и events, снимки памяти

## Подготовка

Нужны [uv](https://docs.astral.sh/uv/) и Python 3.12 (uv поставит сам). Работает на Linux и на macOS с Apple
Silicon; на Windows — через WSL2 (`mtf` использует Unix-модуль `resource`). На macOS окружение занимает ≈1 GB, на
Linux `uv sync` ставит сборку torch с CUDA, это несколько гигабайт.

```bash
uv sync
uv run pytest -q tests/test_train.py tests/test_parallel.py -k "not mine"   # фреймворк и FSDP2 работают
uv run python tools/grade.py                                                # до начала работы: 0 / 8
```

Реализацию FSDP выбирает `--parallel.fsdp_impl`: `fsdp2` (по умолчанию) или `mine` (ваша).
`--parallel.overlap=true` включает часть 2.

```bash
uv run torchrun --master_addr=127.0.0.1 --master_port=29511 --nproc_per_node=4 -m mtf.train \
  --config configs/debug.toml --train.device=cpu --parallel.fsdp_impl=mine
```

Если запуск зависает или падает до первого шага (бывает на macOS), перезапустите его с другим `--master_port`. Потоков на процесс
должно быть не больше, чем ядер: `OMP_NUM_THREADS=<ядра / N>` (torchrun по умолчанию ставит 1).

## Как устроен FSDP в mtf

Цикл обучения — `mtf/train.py`. Модель строится на meta-устройстве (только формы, без памяти), `parallelize`
(`mtf/parallel/__init__.py`) шардирует её, `to_empty` выделяет память под шарды, и в них загружаются веса.
Значит, ваш код шардирования получает параметры на meta и должен работать и там.

Писать код нужно только в `mtf/parallel/my_fsdp.py`, в местах с `NotImplementedError` и `TODO`; остальной код
трогать не нужно. Имена классов, методов и атрибутов оставьте прежними: на них опираются тесты.

- **Юниты.** `apply_fsdp` делает юнит (`FSDPUnit`) из каждого блока `model.layers[i]` и корневой юнит из всей
  модели: в нём остаются эмбеддинг, финальная норма и `lm_head`. Параметры юнита собираются и сводятся вместе.
  При `tie_embeddings` выходной слой — это матрица эмбеддинга, она нужна дважды за forward корня, поэтому они
  в одном юните. Так же юниты устроены в FSDP2 (`mtf/parallel/fsdp.py`).
- **Параметр** (`FSDPParam`) хранит две версии. `sharded` — `nn.Parameter` с DTensor (placement `Shard(0)`):
  строки параметра, которые принадлежат этому рангу. Он лежит в модуле между шагами, его видят оптимизатор,
  клиппинг и чекпоинты. `unsharded` — полный `nn.Parameter`, он лежит в модуле, пока юнит собран. Это один и тот
  же объект на весь запуск: autograd сохраняет его в forward и читает в backward. Освободить его — значит
  сжать storage до 0 байт (`free_unsharded`), объект остаётся; `alloc_unsharded` возвращает память.
- **Хуки** (уже написаны, кроме отмеченных мест) вызывают методы юнита:

| Хук | Когда срабатывает | Что делает |
|---|---|---|
| `pre_forward` | forward pre-hook модуля | `unshard` + `wait_for_unshard`; часть 2: prefetch следующего юнита |
| `post_forward` | forward hook модуля | `reshard`; запоминает порядок юнитов; вешает `pre_backward` на выходы |
| `pre_backward` | готов градиент выхода юнита | `unshard` + `wait_for_unshard`; часть 2: prefetch |
| `post_backward` | готовы градиенты входов юнита (`RegisterPostBackward`) | `reshard` + `reduce_grads` |
| `final_callback` | конец всего backward | `post_backward` тех, у кого он не сработал; часть 2: `finish_reduce` |

Порядок событий за шаг для модели из двух блоков (L0, L1):

```
forward:  pre_forward(root) → эмбеддинг → pre_forward(L0) → L0 → post_forward(L0) → pre_forward(L1) → L1
          → post_forward(L1) → норма, lm_head → post_forward(root)
backward: pre_backward(root) → lm_head, норма → pre_backward(L1) → L1 → post_backward(L1) → pre_backward(L0)
          → L0 → post_backward(L0) → эмбеддинг → final_callback: post_backward(root)
```

У корня `post_backward` срабатывает только в `final_callback`: его входы — номера токенов, градиентов у них нет.

**Коллективы.** Шарды собираются `dist.all_gather_into_tensor`, градиенты сводятся `dist.reduce_scatter_tensor`
(можно и списочные `dist.all_gather` / `dist.reduce_scatter`), группа — `self.group`. Тесты запрещают остальные
коллективы, `DTensor.full_tensor()` / `redistribute()` и `fully_shard` из torch.

## Отладка

Упавший тест показывает ошибку первого упавшего ранга и строки `mtf/`, через которые она прошла. Одну проверку
можно запустить и напрямую, тогда видны полные traceback'и всех рангов и ваши `print`:

```bash
uv run torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29511 tests/dist_checks.py forward
```

Имена проверок: `shard`, `forward`, `backward`, `mixed`, `forward_prefetch`, `backward_prefetch`, `async_reduce`.

| Сообщение | Обычная причина |
|---|---|
| `got mixed torch.Tensor and DTensor` | блок считает на шардах: юнит не собран или полные параметры не подставлены в модуль |
| `... storage of size 0` | backward на освобождённых параметрах: юнит не собран заново в `pre_backward` |
| `Cannot copy out of meta tensor` | использован тензор, запомненный до `to_empty`: берите `p.sharded.to_local()` |
| `... modified by an inplace operation` | `copy_` в `p.unsharded`, а не в `p.unsharded.data` |
| `Check failed: outputShape[0] * worldSize == inputShape[0]` | у `reduce_scatter_tensor` dim 0 входа должен быть в N раз больше dim 0 выхода: проще подавать плоские тензоры |
| `max abs error` в logits | выход all-gather неверно разобран по параметрам: перепутаны куски рангов |
| `max abs error` в градиенте | вход reduce-scatter собран не в том порядке; сумму вместо среднего и перезапись вместо накопления тест называет сам |
| `Timed out waiting ... ms` (через минуту) | ранги вызвали разные коллективы или в разном порядке, например один из них упал раньше |

## Часть 1. FSDP без overlap (5 баллов)

Пока `ctx.overlap` выключен, все коллективы синхронные. Тесты части 1 так и запускают.

**1.1. Шардирование.** В `FSDPParam.__init__` ранг `r` из `N` берёт `r`-й из `N` равных кусков параметра по
dim 0 и оборачивает в DTensor на `mesh` с `Shard(0)`, указав форму и stride полного параметра
(`DTensor.from_local(..., shape=, stride=)`). Шард должен быть копией: view держит в памяти весь параметр.
Размер dim 0 делится на N (это уже проверяется; FSDP2 умеет и неровные шарды, с padding, здесь не нужно).
Параметры приходят на meta, и после `to_empty` у шарда уже другой локальный тензор: не запоминайте его в
`__init__`, а берите `p.sharded.to_local()` там, где он нужен.

**1.2. Forward.** `unshard` собирает шарды всех параметров юнита одним all-gather (можно и по параметру, но
один вызов дешевле) и сохраняет результат в `self.all_gather`. `wait_for_unshard` раскладывает выход по полным
параметрам и подставляет их в модуль, `reshard` освобождает полные параметры и возвращает в модуль шарды.
Разберитесь, как лежат данные в выходе all-gather: кусок от ранга `j` — то, что ранг `j` подал на вход, а шард
ранга `j` — это `j`-й блок строк параметра. Копируйте в `p.unsharded.data`, а не в `p.unsharded`: `copy_` в
параметр увеличит его счётчик версий, и backward откажется от тензоров, которые сохранил в forward.

**1.3. Backward.** Backward тоже нужны полные веса: в `pre_backward` соберите юнит снова. После forward память
полных параметров освобождена, но autograd держит ссылки на те же объекты `unsharded`, и копирование в `.data`
возвращает им данные. В `post_backward` юнит освобождается, и `reduce_grads` сводит градиенты reduce-scatter'ом
в dtype шарда (`p.dtype`), усредняя по рангам. Полные градиенты после этого не нужны (`p.unsharded.grad = None`).
`accumulate_sharded_grads` кладёт результат в `p.sharded.grad` — DTensor, как сам шард. За шаг бывает несколько
micro-batch'ей (gradient accumulation, в тестах их два), и градиенты надо складывать. Подумайте, почему
градиенты надо именно усреднять, а не суммировать.

**1.4. Mixed precision.** С `--train.dtype=bf16` юниты получают `param_dtype=bf16`: полные параметры и всё
вычисление в bf16, шарды приводятся к bf16 до all-gather (передаётся вдвое меньше), градиенты сводятся в fp32,
шарды и состояния оптимизатора остаются в fp32. Входы юнитов к bf16 приводит уже написанный `pre_forward`, как
`MixedPrecisionPolicy(cast_forward_inputs=True)` в FSDP2.

Тесты части 1 (`tests/dist_checks.py`) строят маленькую модель в float64 и её копию с вашим FSDP (как в `mtf`: на
meta, затем `to_empty` и загрузка весов). У каждого ранга своя часть батча, и результат сравнивается с моделью
целиком на всём батче: logits, шарды градиентов после двух micro-batch'ей. Кроме того, проверяется память: пока
считается блок, полные параметры (или ещё не разложенный результат all-gather) держат только он и корень, а после
forward и backward — никто, полных градиентов не остаётся. Без overlap каждый коллектив должен завершиться до
того, как начнёт считаться следующий блок. Тесты обучения запускают 6 шагов
`configs/debug.toml` с N = 4 и N = 8 (со связанными эмбеддингами) и сравнивают loss и норму градиента с одним
процессом (до 1e-5 и 1e-4) и с FSDP2 при том же числе процессов (до 1e-6; верное решение совпадает с FSDP2 бит в бит).

**Критерии:**

| Что | Баллы |
|---|---|
| 1.1: шарды — DTensor с верными строками, без лишней памяти, и на meta (`test_shard`) | 0.5 |
| 1.2: forward совпадает с моделью целиком, полные параметры освобождаются (`test_forward`) | 1.5 |
| 1.3: градиенты совпадают, с накоплением, полные градиенты освобождаются (`test_backward`) | 1 |
| 1.3: обучение повторяет один процесс и FSDP2, N = 4, 8 | 1.5 |
| 1.4: в bf16 logits и градиенты совпадают с FSDP2, all-gather в bf16, reduce-scatter в fp32 | 0.5 |

## Часть 2. Overlap (3 балла)

На GPU FSDP2 прячет коммуникации за вычислениями с помощью CUDA streams: all-gather идёт в отдельном stream, и
пока GPU считает один блок, следующий уже собирается. На CPU streams нет, зато gloo выполняет коллективы в своих
потоках. Вызов с `async_op=True` сразу возвращает `Work`, основной поток продолжает считать, а `work.wait()`
ждёт окончания коллектива. Этим механизмом и надо воспользоваться. Всё в этой части включается флагом
`ctx.overlap` (`--parallel.overlap=true`); без него код должен работать как в части 1.

**2.1. Forward prefetch.** В `pre_forward`, когда свои параметры уже собраны, запустите all-gather юнитов из
`unit.forward_prefetch` (у корня это первый блок, у блока `i` — блок `i + 1`; списки заполняет `apply_fsdp`,
как `set_modules_to_forward_prefetch` в FSDP2). `unshard` с overlap не ждёт коллектив, ждёт `wait_for_unshard`.

**2.2. Backward prefetch.** Backward идёт в обратном порядке forward. `post_forward` записывает юниты в
`ctx.post_forward_order` и номер юнита в этом списке — в `unit.post_forward_index`. `backward_prefetch_target`
возвращает юнит, чей backward идёт следующим, а `pre_backward` запускает его all-gather.

**2.3. Асинхронный reduce-scatter.** С overlap `reduce_grads` не ждёт reduce-scatter, а сохраняет
`(work, output, input)` в `self.reduce`: коллектив идёт, пока считается backward следующего юнита.
`finish_reduce` дожидается его и вызывает `accumulate_sharded_grads`. Одновременно могут идти reduce-scatter'ы
только одного юнита (один вызов на юнит или по вызову на параметр — неважно): прежде чем запустить свои, юнит
дожидается reduce-scatter'ов предыдущего юнита, а reduce-scatter'ов последнего дожидается `final_callback`.
Иначе незавершённые коллективы будут держать входные буферы всех юнитов — полные градиенты всей модели — до
конца backward, и экономия памяти на градиентах пропадёт. Входной буфер
держите в `self.reduce` до `wait()`, как FSDP2: на GPU память, с которой работает коллектив в другом stream,
нельзя раньше времени вернуть аллокатору. gloo держит ссылку на него и сам, поэтому тесты этого не проверяют.

Prefetch держит в памяти сразу два юнита, тест памяти это разрешает, а больше — нет: результат all-gather,
который ждёт своей очереди, тоже занимает память полного юнита. Тесты части 2 по журналу событий
проверяют порядок: all-gather блока `i + 1` в forward (блока `i − 1` в backward) запускается асинхронно до того,
как начинает считаться блок `i`, а ждут его после. Reduce-scatter блока `i` асинхронный, его ждут не раньше,
чем начнётся backward блока `i − 1`, и не позже, чем запустится reduce-scatter следующего юнита. Значения и память проверяются как в
части 1.

**Критерии:**

| Что | Баллы |
|---|---|
| 2.1: forward prefetch, N = 4, 8 | 1 |
| 2.2: backward prefetch, N = 4, 8 | 1 |
| 2.3: асинхронный reduce-scatter; обучение с overlap повторяет один процесс и FSDP2 | 1 |

## Часть 3. Сравнение с FSDP2 (2 балла, проверяется вручную)

`configs/tinystories_fsdp.toml` — модель на 50M параметров с нуля на TinyStories (токены уже лежат в
`data/tinystories.bin`). Половина параметров — эмбеддинг (он же выходной слой, лежит в корневом юните),
половина — 8 блоков. seq_len 512, глобальный батч 8, по одной последовательности на micro-batch при любом N:
активации в каждом процессе одинаковые, и память различается только из-за шардирования. Эталон — 50 шагов в
одном процессе: `reference/tinystories_fsdp_1proc.jsonl` (loss 10.8 → 5.8).

Запустите при N = 4 четыре варианта: FSDP2 и ваш FSDP, каждый без overlap и с ним. Что `--parallel.overlap=true`
делает с FSDP2, написано в `mtf/parallel/fsdp.py`. Пятый прогон — ваш FSDP с overlap при N = 8
(`OMP_NUM_THREADS=1`). Пример:

```bash
OMP_NUM_THREADS=2 uv run torchrun --master_addr=127.0.0.1 --master_port=29511 --nproc_per_node=4 -m mtf.train \
  --config configs/tinystories_fsdp.toml --parallel.fsdp_impl=mine --parallel.overlap=true --train.run_dir=runs/mine_overlap_n4
uv run python tools/compare.py runs/mine_overlap_n4 reference/tinystories_fsdp_1proc.jsonl
uv run python tools/compare.py runs/mine_overlap_n4 runs/fsdp2_n4     # с FSDP2 при том же N
```

На Apple M4 шаг занимает ≈1.5 с в одном процессе и 2–3 с при N = 4 и 8, прогон на 50 шагов — 2–3 минуты.
Время шага на ноутбуке гуляет на 10–20 % от прогона к прогону: выигрыш от overlap надёжнее видно в профиле, по
времени, которое основной поток проводит в хуках FSDP. У верного решения расхождение с эталоном — порядка 1e-7
по loss и 1e-6 по норме градиента на всех 50 шагах, а с FSDP2 при том же N — ноль. Эталон посчитан на
Apple M4; на другом процессоре расхождение с ним может быть больше (другой порядок суммирования в fp32), но с
FSDP2, запущенным на той же машине, верное решение всё равно совпадает бит в бит.

Профиль одного шага пишется, если добавить `--train.steps=6 --profile.enabled=true --profile.wait=3
--profile.warmup=1 --profile.active=1`: в `<run_dir>/profile/` появятся трейс каждого ранга (`trace_rank{R}.json`,
открывается в https://ui.perfetto.dev) и таблица самых долгих операций (`ops_rank{R}.txt`). В трейсе видны хуки
обеих реализаций (`FSDP::pre_forward (layers.3)` и т. п.) и коллективы gloo (`gloo:all_gather`) в потоках gloo.

Отчёт — 1–2 страницы:

1. **Таблица** по пяти прогонам: время шага (медиана `step_s` из `metrics.jsonl` без первого шага), память на
   ранг (`mem_gb`: пиковый RSS процесса), расхождение loss и нормы градиента с эталоном в шагах 1–40 и с FSDP2 при
   том же N (`tools/compare.py`). Почему на одном CPU FSDP не ускоряет шаг по сравнению с одним процессом?
2. **Профиль.** Скриншоты forward и backward одного блока из трейсов вашего FSDP без overlap и с ним и FSDP2: где
   идёт all-gather и ждёт ли его основной поток. Сколько all-gather'ов и reduce-scatter'ов за шаг у вашего FSDP и
   у FSDP2? Если числа разные, объясните почему (посмотрите `reshard_after_forward` в документации `fully_shard`).
   Ещё два `gloo:all_reduce` за шаг — не от FSDP: это среднее loss для лога и норма градиента в клиппинге.
3. **Память.** Оцените на бумаге, сколько занимают на ранг шарды весов, градиентов и состояний AdamW, полный
   корневой юнит и полный блок с их градиентами при N = 1, 4, 8, и сравните с измеренным `mem_gb` (сам процесс
   с torch и gloo занимает ≈0.35 GB: столько показывает `configs/debug.toml`). Почему память на ранг падает
   гораздо меньше, чем в N раз? `mem_gb` одной и той же конфигурации гуляет на ≈0.1 GB от прогона к прогону:
   с оценкой сравнивайте крупные различия (N = 1, 4, 8), а не FSDP2 против своего FSDP при одном N.
4. **Вопросы.** (а) Что изменится в числе all-gather'ов и в памяти, если не освобождать параметры после forward
   (`reshard_after_forward=False`)? (б) Почему FSDP2 на GPU прячет all-gather за вычислениями без явного prefetch,
   а на CPU нет? Что меняет `--parallel.overlap=true` для FSDP2 и почему его reduce-scatter всё равно блокирует
   цикл обучения? (в) В трейсе reduce-scatter на gloo выглядит как `gloo:all_reduce`. Что это значит для объёма
   пересылок и как это устроено в NCCL?

**Критерии:**

| Что | Баллы |
|---|---|
| таблица по пяти прогонам с выводами `compare.py`, объяснение времени шага | 0.5 |
| профили: overlap виден на скриншотах, число коллективов за шаг объяснено | 0.5 |
| оценка памяти против измерений, ответ, почему не в N раз | 0.5 |
| ответы на вопросы (а)–(в) | 0.5 |

## Проверка

```bash
uv run pytest -q -m part1            # тесты одной части (part1, part2)
uv run python tools/grade.py         # все тесты и баллы автоматической части, ≈3 минуты
```

## Что сдать

Архив с `mtf/parallel/my_fsdp.py`, выводом `tools/grade.py` и отчётом по части 3 (`report.md` или PDF) с
таблицей, скриншотами профилей и файлами `metrics.jsonl` прогонов.

Задание выдаётся впервые: если что-то непонятно или сломано, пишите в чат курса.
