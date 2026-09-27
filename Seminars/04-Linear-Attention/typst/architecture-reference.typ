#let horizontalrule = line(start: (25%,0%), end: (75%,0%))

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

#set table(
  inset: 6pt,
  stroke: none
)

#show figure.where(
  kind: table
): set figure.caption(position: top)

#show figure.where(
  kind: image
): set figure.caption(position: bottom)

#let content-to-string(content) = {
  if content.has("text") {
    content.text
  } else if content.has("children") {
    content.children.map(content-to-string).join("")
  } else if content.has("body") {
    content-to-string(content.body)
  } else if content == [ ] {
    " "
  }
}
#let conf(
  title: none,
  subtitle: none,
  authors: (),
  keywords: (),
  date: none,
  abstract: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: (),
  fontsize: 11pt,
  sectionnumbering: none,
  pagenumbering: "1",
  doc,
) = {
  set document(
    title: title,
    author: authors.map(author => content-to-string(author.name)),
    keywords: keywords,
  )
  set page(
    paper: paper,
    margin: margin,
    numbering: pagenumbering,
    columns: cols,
    )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)

  place(top, float: true, scope: "parent", clearance: 4mm)[
  #if title != none {
    align(center)[#block(inset: 2em)[
      #text(weight: "bold", size: 1.5em)[#title]
      #(if subtitle != none {
        parbreak()
        text(weight: "bold", size: 1.25em)[#subtitle]
      })
    ]]
  }

  #if authors != none and authors != [] {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  #if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  #if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[Abstract] #h(1em) #abstract
    ]
  }
  ]

  doc
}
#show: doc => conf(
  lang: "ru",
  region: "RU",
  margin: (x: 2.2cm,y: 2.2cm,),
  paper: "a4",
  font: ("Libertinus Serif",),
  fontsize: 10.5pt,
  pagenumbering: "1",
  cols: 1,
  doc,
)

// Оформление: код, таблицы, рисунки, ссылки.
#show raw: set text(font: "DejaVu Sans Mono", size: 0.85em)
#show raw.where(block: true): block.with(fill: luma(245), inset: 6pt, radius: 2pt, width: 100%)
#set table(stroke: 0.4pt + luma(170), inset: 4pt)
#show table: set text(size: 0.85em)
#show table: set par(justify: false)
// pandoc оборачивает и рисунки, и таблицы в figure: рисунок не разрывается,
// а длинная таблица переносится на следующую страницу
#show figure.where(kind: image): set block(breakable: false)
#show figure.where(kind: table): set block(breakable: true)
#show link: set text(fill: rgb("#1a4f8b"))


= Справочник по архитектурам (раздаточный материал)
<справочник-по-архитектурам-раздаточный-материал>
#emph[Справочный материал к модулю про линейное внимание: два числа состояния для девяти архитектур, конфиги, по которым они посчитаны, и источники.]

== Два числа
<два-числа>
Каждой архитектуре в этом модуле сопоставлены два числа; они определены один раз и везде используются под этими названиями и никакими другими.

#strong[Байты состояния на слой при длине контекста `N`.] Сколько памяти должен удерживать один слой после `N` токенов, чтобы выдать следующий токен.

- Softmax-внимание, на слой: `2 * n_kv_heads * head_dim * dtype_bytes * N` байт --- #strong[линейно по `N`];.
- Каждая архитектура линейного внимания, на слой: `H * d_k * d_v * dtype_bytes` байт --- #strong[не зависит от `N`];.
- Гибриды --- сумма по их слоям.

#strong[Прирост байтов на токен, на слой.] На сколько это число растёт, когда приходит ещё один токен.

- Softmax-внимание: `2 * n_kv_heads * head_dim * dtype_bytes` байт на токен --- #strong[константа, и она ненулевая];.
- Каждая архитектура линейного внимания: #strong[`0` байт на токен.]

#quote(block: true)[
Формулировка #strong[«байты состояния на токен» запрещена.] Для softmax-внимания это число-прирост, а для всего остального --- бессмысленная величина вида `1/N`, и одна подпись на две разные единицы --- ровно то, из-за чего контраст ниже перестаёт сходиться. Каждый раз говорите, которое из двух чисел вы имеете в виду.
]

Ненулевой прирост против нулевого --- вот предмет этого модуля.

== Разобранные эталонные числа
<разобранные-эталонные-числа>
Эталонная модель лекции, пересчитанная заново: #strong[по образцу Llama-3-8B: 32 слоя, `n_kv_heads = 8`, `head_dim = 128`, fp16];, против линейного слоя той же ширины (`H = 32`, `d_k = d_v = 128`, fp16).

- Softmax --- прирост байтов на токен, на слой: #strong[4 KiB (4,096 B)]
- Softmax --- байты состояния на слой при `N` = 128k: #strong[512 MiB (536,870,912 B)]
- Softmax --- состояние по всей модели при `N` = 128k (32 слоя): #strong[16 GiB (17,179,869,184 B)]
- Линейный слой --- байты состояния на слой, при любом `N`: #strong[1 MiB (1,048,576 B)]
- Линейный слой --- состояние по всей модели (32 слоя), при любом `N`: #strong[32 MiB (33,554,432 B)]
- Линейный слой --- прирост байтов на токен, на слой: #strong[0 B]
- Разрыв на слой при `N` = 128k: #strong[512x]
- Ловушка `n_heads` (32 KV-головы вместо 8): #strong[64 GiB (68,719,476,736 B)]

Разрыв --- `512x` на слой при 128k, и он не ограничен по `N`: число линейного слоя не двигается с места.

== Таблица
<таблица>
Длины контекста: `4k` = 4,096, `32k` = 32,768, `128k` = 131,072. Каждая строка посчитана при `dtype_bytes = 2`; конфиги и типы данных каждой строки --- в следующей таблице.

#figure(
  align(center)[#table(
    columns: 6,
    align: (auto,auto,auto,auto,auto,auto,),
    table.header([Архитектура], [Форма затухания], [Правило обновления], [Нормализация], [Байты состояния на слой при `N`], [Прирост байтов на токен, на слой],),
    table.hline(),
    [#strong[Softmax-внимание (эталон)];], [нет --- кэш точный, ничего не забывается], [дописать (k\_t, v\_t) в кэш; o\_t = softmax(q\_t K^T / sqrt(d\_k)) V], [softmax по оценкам внимания, с масштабированием на 1/sqrt(d\_k)], [#strong[16 MiB при 4k / 128 MiB при 32k / 512 MiB при 128k] (GQA self-attention, линейно по `N`)], [#strong[4 KiB (4,096 B)] (GQA self-attention)],
    [#strong[Linear Transformer];#footnote[#strong[Linear Transformer --- конфиг не привязан к выпущенной модели.] Оригинальная статья не публикует конфигурации масштаба языковой модели. Её самая большая модель --- 16-слойный 8-головый авторегрессионный трансформер для изображений CIFAR-10 с размером эмбеддинга 256, то есть d\_k = d\_v = 32 на голову (arXiv 2006.16236, раздел 4.2.2, который переиспользует конфигурацию MNIST из раздела 4.2.1), и dtype в ней не указан. Чекпоинта `linear_attn` на fla-hub нет --- на практике семейство вытеснено своими потомками с гейтами. Поэтому числа в этой строке вычислены по значениям по умолчанию класса `LinearAttentionConfig` из flash-linear-attention при форме 1.3B, общей для всех остальных строк FLA, и читать их нужно как «сколько эта архитектура стоила бы при такой ширине», а не как измерение обученной модели.];], [нет (alpha = 1); состояние только растёт], [аддитивное: S\_t = S\_{t-1} + phi(k\_t) v\_t^T], [знаменатель sum-phi(k) (накопительный вектор z\_t), отображение признаков phi], [#strong[2 MiB] при любом `N` (линейное внимание)], [#strong[0 B] (линейное внимание)],
    [#strong[RetNet];], [фиксированный скаляр на голову, gamma\_h --- не зависит от данных, не выучивается на каждый токен], [аддитивное с гейтом: S\_t = gamma\_h S\_{t-1} + k\_t v\_t^T], [GroupNorm/RMSNorm на выходе по головам, плюс выходной гейт со swish; знаменателя sum-phi(k) нет], [#strong[2 MiB] при любом `N` (retention)], [#strong[0 B] (retention)],
    [#strong[GLA];], [поканальный вектор, зависящий от входа: Diag(alpha\_t), по гейту на каждый канал ключа], [аддитивное с гейтом: S\_t = Diag(alpha\_t) S\_{t-1} + k\_t v\_t^T], [RMSNorm на выходе плюс выходной гейт (use\_output\_gate = true); знаменателя нет], [#strong[1 MiB] при любом `N` (линейное внимание с гейтом)], [#strong[0 B] (линейное внимание с гейтом)],
    [#strong[Mamba2];], [зависящий от входа скаляр на голову: alpha\_t = exp(-Delta\_t \* a\_h)], [аддитивное с гейтом (SSD): S\_t = alpha\_t S\_{t-1} + k\_t v\_t^T, где k\_t = B\_t, v\_t = x\_t], [RMSNorm с гейтом на выходе (rmsnorm = True), плюс короткая depthwise-свёртка на входе; знаменателя нет], [#strong[1.25 MiB] при любом `N` (Mamba2 SSD)], [#strong[0 B] (Mamba2 SSD)],
    [#strong[DeltaNet];], [нет --- члена затухания нет вовсе], [дельта-правило: S\_t = (I --- beta\_t k\_t k\_t^T) S\_{t-1} + beta\_t k\_t v\_t^T (стирающий множитель слева)], [L2-нормированные q и k (qk\_norm = \'l2\') после SiLU, короткая depthwise-свёртка, RMSNorm на выходе; выходного гейта нет (use\_gate = false)], [#strong[512 KiB] при любом `N` (дельта-правило)], [#strong[0 B] (дельта-правило)],
    [#strong[Gated DeltaNet];], [зависящий от входа скаляр на голову, alpha\_t (в стиле Mamba2, из параметризации через softplus/exp)], [дельта-правило с гейтом: S\_t = alpha\_t (I --- beta\_t k\_t k\_t^T) S\_{t-1} + beta\_t k\_t v\_t^T], [L2-нормированные q/k после SiLU, короткая depthwise-свёртка на q/k/v, RMSNorm с гейтом на выходе], [#strong[1.37 MiB] при любом `N` (дельта-правило с гейтом)], [#strong[0 B] (дельта-правило с гейтом)],
    [#strong[Gated DeltaNet-2];#footnote[#strong[Gated DeltaNet-2 --- тип данных состояния.] Приложение D.3: \'The recurrent state is stored in fp32 across chunks and during recurrent decoding.\' При fp32 состояние этой строки на слой --- 1.00 MiB, а не 512 KiB. В таблице эта строка, как и все остальные, посчитана по 2 байта, чтобы сравнивалась ширина; в реальном развёртывании увидят именно цифру для fp32.];], [поканальный вектор alpha\_t = exp(g\_t), g\_t = -exp(a) \* softplus(W\_f x\_t + delta)], [дельта-правило-2 с гейтом (Gated Delta Rule-2), стирание и запись разделены: S\_t = (I --- k\_t (b\_t . k\_t)^T) Diag(alpha\_t) S\_{t-1} + k\_t (w\_t . v\_t)^T, с поканальным гейтом стирания b\_t in \[0,1\]^{d\_k} и поканальным гейтом записи w\_t in \[0,1\]^{d\_v}], [L2-нормированные q/k после короткой свёртки + SiLU, RMSNorm на выходе с выходным гейтом на SiLU], [#strong[512 KiB] при любом `N` (дельта-правило-2 с гейтом)], [#strong[0 B] (дельта-правило-2 с гейтом)],
    [#strong[KDA (Kimi Linear)];#footnote[Гибридная строка --- см. раздел #emph[Две гибридные строки, разложенные явно] выше.];#footnote[#strong[KDA (Kimi Linear) --- член, линейный по `N`, здесь MLA, а не GQA.] Член, линейный по N, здесь --- MLA, который НЕ подчиняется GQA-формуле из раздела «Два числа». MLA кэширует один сжатый латент на токен на слой, шириной kv\_lora\_rank + qk\_rope\_head\_dim = 512 + 64 = 576, общий для всех 32 голов: 1152 байта на токен на слой MLA. Именно на таком учёте держится заявление о 75%. Эталонная HF-реализация (modeling\_kimi.py, KimiMLAAttention.forward) вместо этого материализует в кэш развёрнутые K и V по головам --- 32 \* (192 + 128) = 10,240 элементов, 20,480 байт на токен на слой, \~18x больше. Эффективный \'absorbed\'-путь, который используется в vLLM, кэширует латент. Цифра 75% определяется соотношением слоёв и одинакова при любом из двух учётов.];], [поканальное (тонкозернистое) диагональное затухание Diag(alpha\_t), переход DPLR с ограничениями; у слоёв полного внимания затухания нет], [дельта-правило с гейтом и поканальным затуханием (KDA) на 20 слоях; softmax-MLA на 7 слоях], [L2-нормированные q/k, короткая свёртка (ядро 4), RMSNorm на выходе с сигмоидным выходным гейтом; NoPE на всех слоях MLA], [#strong[1 MiB] при любом `N` (KDA (линейное))+ #strong[4.50 MiB при 4k / 36 MiB при 32k / 144 MiB при 128k] (полное внимание MLA, линейно по `N`)], [#strong[0 B] (KDA (линейное))+ #strong[1,152 B] (полное внимание MLA)],
    [#strong[Qwen3-Next];#footnote[Гибридная строка --- см. раздел #emph[Две гибридные строки, разложенные явно] выше.];], [зависящий от входа скаляр на голову значений, g\_t = -exp(A\_log) \* softplus(a\_t + dt\_bias); у слоёв полного внимания затухания нет], [дельта-правило с гейтом (Qwen3NextGatedDeltaNet) на 36 слоях; softmax-внимание с GQA на 12 слоях], [L2-нормированные q/k внутри ядра, короткая depthwise-свёртка (ядро 4), RMSNorm с гейтом на выходе; слои внимания используют частичный RoPE (25%) и выход с гейтом], [#strong[1 MiB] при любом `N` (Gated DeltaNet (линейное))+ #strong[8 MiB при 4k / 64 MiB при 32k / 256 MiB при 128k] (полное внимание GQA, линейно по `N`)], [#strong[0 B] (Gated DeltaNet (линейное))+ #strong[2 KiB (2,048 B)] (полное внимание GQA)],
  )]
  , kind: table
  )

== Откуда взяты числа каждой строки
<откуда-взяты-числа-каждой-строки>
Каждая строка называет конфиг, по которому она вычислена, и свой `dtype_bytes`. Столбцы «по всей модели» --- это числа на слой, просуммированные по слоям модели: то, что блок 0 лекции считает вживую и что нужно рассуждению о гибридах в блоке 6.

#figure(
  align(center)[#table(
    columns: 7,
    align: (auto,auto,auto,auto,auto,auto,auto,),
    table.header([Архитектура], [Конфиг, по которому вычислено], [`dtype_bytes`], [Слои], [Формула на слой], [Состояние по всей модели при `4k` / `32k` / `128k`], [Прирост по всей модели],),
    table.hline(),
    [#strong[Softmax-внимание (эталон)];], [Llama-3-8B (meta-llama/Meta-Llama-3-8B) (опубликованный чекпоинт)], [2], [32 × GQA self-attention], [`2*n_kv_heads*head_dim*dtype_bytes*N = 2*8*128*2*N`], [512 MiB / 4 GiB / 16 GiB], [128 KiB],
    [#strong[Linear Transformer];], [flash-linear-attention LinearAttentionConfig, значения по умолчанию (форма 1.3B) (эталонный конфиг из библиотеки --- НЕ выпущенный чекпоинт)], [2], [24 × линейное внимание], [`H*d_k*d_v*dtype_bytes = 4*512*512*2`], [48 MiB / 48 MiB / 48 MiB], [#strong[0 B];],
    [#strong[RetNet];], [fla-hub/retnet-1.3B-100B (опубликованный чекпоинт)], [2], [24 × retention], [`H*d_k*d_v*dtype_bytes = 8*256*512*2`], [48 MiB / 48 MiB / 48 MiB], [#strong[0 B];],
    [#strong[GLA];], [fla-hub/gla-1.3B-100B (опубликованный чекпоинт)], [2], [24 × линейное внимание с гейтом], [`H*d_k*d_v*dtype_bytes = 4*256*512*2`], [24 MiB / 24 MiB / 24 MiB], [#strong[0 B];],
    [#strong[Mamba2];], [state-spaces/mamba2-2.7b (опубликованный чекпоинт)], [2], [64 × Mamba2 SSD], [`H*d_k*d_v*dtype_bytes = 80*128*64*2`], [80 MiB / 80 MiB / 80 MiB], [#strong[0 B];],
    [#strong[DeltaNet];], [fla-hub/delta\_net-1.3B-100B (опубликованный чекпоинт)], [2], [24 × дельта-правило], [`H*d_k*d_v*dtype_bytes = 16*128*128*2`], [12 MiB / 12 MiB / 12 MiB], [#strong[0 B];],
    [#strong[Gated DeltaNet];], [NVlabs/GatedDeltaNet, конфиг `GatedDeltaNet_1.3B` (опубликованный конфиг обучения (официальный релиз кода; чекпоинта на HF нет))], [2], [16 × дельта-правило с гейтом], [`H*d_k*d_v*dtype_bytes = 9*200*400*2`], [21.97 MiB / 21.97 MiB / 21.97 MiB], [#strong[0 B];],
    [#strong[Gated DeltaNet-2];], [NVlabs/GatedDeltaNet-2, конфиг `gdn2_1.3B`; размерности голов сверены с приложением E.1 arXiv 2605.22791 (опубликованный конфиг обучения (официальный релиз кода; чекпоинта на HF нет))], [2], [18 × дельта-правило-2 с гейтом], [`H*d_k*d_v*dtype_bytes = 16*128*128*2`], [9 MiB / 9 MiB / 9 MiB], [#strong[0 B];],
    [#strong[KDA (Kimi Linear)];], [moonshotai/Kimi-Linear-48B-A3B-Instruct (48B всего, 3B активных, MoE) (опубликованный чекпоинт)], [2], [20 × KDA (линейное) + 7 × полное внимание MLA], [`H*d_k*d_v*dtype_bytes = 32*128*128*2``latent_dim*dtype_bytes*N = 576*2*N`], [51.50 MiB / 272 MiB / 1.00 GiB], [8,064 B],
    [#strong[Qwen3-Next];], [Qwen/Qwen3-Next-80B-A3B-Instruct (80B всего, 3B активных, MoE) (опубликованный чекпоинт)], [2], [36 × Gated DeltaNet (линейное) + 12 × полное внимание GQA], [`H*d_k*d_v*dtype_bytes = 32*128*128*2``2*n_kv_heads*head_dim*dtype_bytes*N = 2*2*256*2*N`], [132 MiB / 804 MiB / 3.04 GiB], [24 KiB],
  )]
  , kind: table
  )

== Две гибридные строки, разложенные явно
<две-гибридные-строки-разложенные-явно>
Гибрид --- это ровно та строка, которую нельзя описать одним столбцом: #strong[постоянный член] от его линейных слоёв плюс #strong[член, линейный по `N`];, от его слоёв полного внимания. Здесь оба выписаны отдельно, а не свёрнуты в сумму.

=== KDA (Kimi Linear) --- moonshotai/Kimi-Linear-48B-A3B-Instruct (48B всего, 3B активных, MoE)
<kda-kimi-linear--moonshotaikimi-linear-48b-a3b-instruct-48b-всего-3b-активных-moe>
Соотношение, заявленное в статье: #strong[3:1 (KDA : полное MLA)];. Соотношение в выпущенном чекпоинте: #strong[20:7 = 2.86:1 на 27 слоях];.

- #strong[Постоянный член] --- 20 × KDA (линейное): `H*d_k*d_v*dtype_bytes = 32*128*128*2` = #strong[1 MiB (1,048,576 B) на слой];, при любом `N`, #strong[20 MiB по всей модели];. Прирост: #strong[0 B на токен];.

- #strong[Член, линейный по `N`] --- 7 × полное внимание MLA: прирост #strong[1,152 B на токен на слой];, то есть 4.50 MiB при 4k, 36 MiB при 32k, 144 MiB при 128k на слой (31.50 MiB при 4k, 252 MiB при 32k, 1008 MiB при 128k по всей модели).

- #strong[Прирост по всей модели: 8,064 B на токен] --- целиком от 7 слоёв полного внимания.

- #strong[Заявление о сокращении KV-кэша.] Базовый вариант (для заявленных 75% сравниваем с полным MLA на всех 27 слоях): `27 * 1,152 B` = #strong[31,104 B на токен];. Эта модель: #strong[8,064 B на токен];. Сокращение = #strong[74.1%];. Это утверждение о числе-#emph[приросте] и ни о чём другом --- постоянный член от линейных слоёв им не затрагивается, и никакой «размер состояния на токен» здесь не заявляется.

=== Qwen3-Next --- Qwen/Qwen3-Next-80B-A3B-Instruct (80B всего, 3B активных, MoE)
<qwen3-next--qwenqwen3-next-80b-a3b-instruct-80b-всего-3b-активных-moe>
Соотношение, заявленное в статье: #strong[3:1 (линейное : полное внимание)];. Соотношение в выпущенном чекпоинте: #strong[36:12 = ровно 3:1];.

- #strong[Постоянный член] --- 36 × Gated DeltaNet (линейное): `H*d_k*d_v*dtype_bytes = 32*128*128*2` = #strong[1 MiB (1,048,576 B) на слой];, при любом `N`, #strong[36 MiB по всей модели];. Прирост: #strong[0 B на токен];.

- #strong[Член, линейный по `N`] --- 12 × полное внимание GQA: прирост #strong[2 KiB (2,048 B) на токен на слой];, то есть 8 MiB при 4k, 64 MiB при 32k, 256 MiB при 128k на слой (96 MiB при 4k, 768 MiB при 32k, 3 GiB при 128k по всей модели).

- #strong[Прирост по всей модели: 24 KiB (24,576 B) на токен] --- целиком от 12 слоёв полного внимания.

- #strong[Заявление о сокращении KV-кэша.] Базовый вариант (для заявленных 75% сравниваем с полным вниманием (GQA) на всех 48 слоях): `48 * 2,048 B` = #strong[96 KiB (98,304 B) на токен];. Эта модель: #strong[24 KiB (24,576 B) на токен];. Сокращение = #strong[75.0%];. Это утверждение о числе-#emph[приросте] и ни о чём другом --- постоянный член от линейных слоёв им не затрагивается, и никакой «размер состояния на токен» здесь не заявляется.

== Примечания и ссылки на источники
<примечания-и-ссылки-на-источники>
=== Проверка цитирований и утверждений
<проверка-цитирований-и-утверждений>
Каждое число, взятое из источника, один раз сверено с этим источником.

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([Утверждение], [Вердикт], [Что на самом деле говорит источник],),
    table.hline(),
    [arXiv 2507.19595 --- это #emph[Efficient Attention Mechanisms for LLMs: A Survey];], [#strong[разрешается; название сокращено];], [Efficient Attention Mechanisms for Large Language Models: A Survey --- Yutao Sun, Zhenyu Li, Yike Zhang, Tengyu Pan, Bowen Dong, Yuyi Guo, Jianyong Wang; submitted 2025-07-25],
    [arXiv 2406.06484 --- это #emph[Parallelizing Linear Transformers with the Delta Rule over Sequence Length];], [#strong[разрешается точно];], [Parallelizing Linear Transformers with the Delta Rule over Sequence Length --- Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim; submitted 2024-06-10],
    [arXiv 2510.26692 --- это #emph[Kimi Linear];], [#strong[разрешается точно];], [Kimi Linear: An Expressive, Efficient Attention Architecture --- Kimi Team (Yu Zhang, Zongyu Lin, Xingcheng Yao, ... , Yulun Du); submitted 2025-10-30],
    [arXiv 2605.22791 --- это #emph[Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention];], [#strong[подтверждено: статья существует; идентификатор, название и авторство совпадают];], [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention --- Ali Hatamizadeh, Yejin Choi, Jan Kautz (NVIDIA); arXiv:2605.22791v1 \[cs.AI\], submitted 2026-05-21. Code: #link("https://github.com/NVlabs/GatedDeltaNet-2") (HTTP 200 on 2026-09-05; it carries an official `gdn2_1.3B` training config and the GDN-2 layer implementation, which is where this row\'s layer count comes from).],
    [Сокращение KV-кэша на 75% у Kimi Linear --- утверждение о числе-#emph[приросте];, а не о каком-либо размере состояния на токен], [#strong[подтверждено, и арифметика сходится];], [The paper\'s wording is \'reduces memory and KV-cache usage by up to 75% during long-sequence generation\' (introduction) and \'reducing KV cache usage by up to 75%\' (abstract). KV-cache usage is the quantity that grows with sequence length, i.e. the marginal number. It closes exactly as a layer ratio: at a clean 3:1 only one layer in four carries an N-linear cache, giving 25% of full attention and a 75% reduction. The shipped 27-layer checkpoint is 20:7, giving 8,064 vs 31,104 bytes per token = 74.1%. Nothing in the claim concerns the constant per-layer KDA state, which does not move with N at all.],
    [Гибридное соотношение Kimi Linear --- 3:1], [#strong[3:1 в статье; 20:7 (2.86:1) в выпущенном чекпоинте];], [Paper: \'Kimi Linear interleaves KDA with periodic full attention layers in a uniform 3:1 ratio\'; the section-5.2 ablation puts 3:1 at training ppl 9.23 / validation 5.65, against 0:1, 1:1, 7:1 and 15:1. Checkpoint: moonshotai/Kimi-Linear-48B-A3B-Instruct has num\_hidden\_layers = 27 with full\_attn\_layers = \[4, 8, 12, 16, 20, 24, 27\] --- the 3:1 pattern over the first 24 layers, plus a seventh full-attention layer at the top. Teach 3:1; the handout prints 20:7 because that is what ships.],
    [Гибридное соотношение Qwen3-Next --- 3:1, а его линейные слои --- Gated DeltaNet при H=32, d\_k=d\_v=128], [#strong[подтверждено по исходникам HF и опубликованному конфигу];], [transformers configuration\_qwen3\_next.py builds layer\_types as \[\'linear\_attention\' if bool((i + 1) % full\_attention\_interval) else \'full\_attention\'\], and the published config sets full\_attention\_interval = 4 with num\_hidden\_layers = 48: 12 full-attention layers, 36 linear, exactly 3:1. linear\_num\_key\_heads = 16 but linear\_num\_value\_heads = 32, and modular\_qwen3\_next.py repeat\_interleave\'s q and k by 2 before the recurrence, so the state carries 32 heads of 128 x 128 --- not 16. That is the reference linear layer of the lecture, to the byte.],
    [Числа лекции для Llama-3-8B и эталонного линейного слоя: прирост 4 KiB/токен/слой, 512 MiB на слой при 128k, 16 GiB по всей модели, 1 MiB и 32 MiB для линейного слоя, разрыв 512x на слой при 128k], [#strong[все восемь пересчитаны, и все восемь сходятся];], [Пересчитаны в разделе #emph[Разобранные эталонные числа] выше из тех же констант, что и таблица.],
  )]
  , kind: table
  )

=== Подробности по строкам
<подробности-по-строкам>
==== Softmax-внимание (эталон)
<softmax-внимание-эталон>
- #strong[Конфиг:] Llama-3-8B (meta-llama/Meta-Llama-3-8B) (опубликованный чекпоинт).
- #strong[`dtype_bytes` = 2] --- config.json torch\_dtype = bfloat16; в лекции --- fp16. Те же 2 байта в любом случае.
- #strong[Статья:] arXiv 2407.21783 --- The Llama 3 Herd of Models
- #strong[Источник конфига:] #link("https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json")
- #strong[Зеркало конфига] (канонический репозиторий требует авторизации): #link("https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json")
- #strong[Примечание:] GQA, а не MHA: num\_attention\_heads = 32, но num\_key\_value\_heads = 8. Кэшируются только KV-головы. Если подставить сюда n\_heads, получится 64 GiB по всей модели при 128k вместо правильных 16 GiB --- та самая ошибка, от которой предостерегает раздел 0 лекции.
- #strong[Ссылки на источники:]
  - num\_hidden\_layers = 32, num\_attention\_heads = 32, num\_key\_value\_heads = 8, hidden\_size = 4096, torch\_dtype = bfloat16 --- Llama-3-8B config.json. Канонический репозиторий meta-llama/Meta-Llama-3-8B требует авторизации (HTTP 401 без учётных данных), поэтому значения считаны 2026-09-05 с открытого зеркала NousResearch/Meta-Llama-3-8B; это широко опубликованные числа Llama-3-8B, и они совпадают с разделом 0 лекции.
  - head\_dim = hidden\_size / num\_attention\_heads = 4096 / 32 = 128 (в этой версии у LlamaConfig нет поля head\_dim; transformers выводит его сам)

==== Linear Transformer
<linear-transformer>
- #strong[Конфиг:] flash-linear-attention LinearAttentionConfig, значения по умолчанию (форма 1.3B) (эталонный конфиг из библиотеки --- НЕ выпущенный чекпоинт).
- #strong[`dtype_bytes` = 2] --- не опубликован: класс конфига не несёт dtype; принято 2 байта, как у каждого чекпоинта fla-hub на 1.3B (torch\_dtype = bfloat16).
- #strong[Статья:] arXiv 2006.16236 --- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos, Vyas, Pappas, Fleuret, ICML 2020)
- #strong[Источник конфига:] #link("https://github.com/fla-org/flash-linear-attention/blob/64efbae863fdaf76fa5359e517b31cf1d72525a1/fla/models/linear_attn/configuration_linear_attn.py")
- #strong[Примечание:] `feature_map = 'elementwise_product'` (HadamardFeatureMap) сохраняет размерность головы, так что размерность признаков равна d\_k и состояние остаётся d\_k x d\_v.
- #strong[НЕ ПРИВЯЗАНО К ВЫПУЩЕННОЙ МОДЕЛИ:] Оригинальная статья не публикует конфигурации масштаба языковой модели. Её самая большая модель --- 16-слойный 8-головый авторегрессионный трансформер для изображений CIFAR-10 с размером эмбеддинга 256, то есть d\_k = d\_v = 32 на голову (arXiv 2006.16236, раздел 4.2.2, который переиспользует конфигурацию MNIST из раздела 4.2.1), и dtype в ней не указан. Чекпоинта `linear_attn` на fla-hub нет --- на практике семейство вытеснено своими потомками с гейтами. Поэтому числа в этой строке вычислены по значениям по умолчанию класса `LinearAttentionConfig` из flash-linear-attention при форме 1.3B, общей для всех остальных строк FLA, и читать их нужно как «сколько эта архитектура стоила бы при такой ширине», а не как измерение обученной модели.
- #strong[Не учтено в столбце состояния:] нормализатор sum-phi(k), вектор z\_t (H \* d\_k элементов) --- ещё 4 KiB (4,096 B) на каждый слой типа «линейное внимание», и это не зависит от `N`. Формула из раздела «Два числа» этого не включает.
- #strong[Ссылки на источники:]
  - hidden\_size = 2048, num\_hidden\_layers = 24, num\_heads = 4, expand\_k = 1.0, expand\_v = 1.0, feature\_map = \'elementwise\_product\' --- fla/models/linear\_attn/configuration\_linear\_attn.py, значения по умолчанию в LinearAttentionConfig.#strong[init]
  - d\_k = hidden\_size \* expand\_k / num\_heads = 512, d\_v = hidden\_size \* expand\_v / num\_heads = 512 --- fla/layers/linear\_attn.py, строки 56-68
  - самый большой конфиг оригинальной статьи: 16 слоёв, 8 голов, эмбеддинг 256 (32 измерения на голову) --- arXiv 2006.16236, разделы 4.2.1-4.2.2

==== RetNet
<retnet>
- #strong[Конфиг:] fla-hub/retnet-1.3B-100B (опубликованный чекпоинт).
- #strong[`dtype_bytes` = 2] --- config.json torch\_dtype = bfloat16.
- #strong[Статья:] arXiv 2307.08621 --- Retentive Network: A Successor to Transformer for Large Language Models
- #strong[Источник конфига:] #link("https://huggingface.co/fla-hub/retnet-1.3B-100B/raw/main/config.json")
- #strong[Примечание:] Голова значений вдвое шире головы ключей: expand\_v = 2 против expand\_k = 1 --- это выбор самой статьи и причина, по которой состояние в этой строке 2 MiB, а не 1 MiB.
- #strong[Ссылки на источники:]
  - hidden\_size = 2048, num\_hidden\_layers = 24, num\_heads = 8, expand\_k = 1, expand\_v = 2, use\_output\_gate = true, torch\_dtype = bfloat16 --- fla-hub/retnet-1.3B-100B config.json
  - d\_k = hidden\_size \* expand\_k / num\_heads = 256, d\_v = hidden\_size \* expand\_v / num\_heads = 512 --- fla/layers/multiscale\_retention.py, строки 108-121

==== GLA
<gla>
- #strong[Конфиг:] fla-hub/gla-1.3B-100B (опубликованный чекпоинт).
- #strong[`dtype_bytes` = 2] --- config.json torch\_dtype = bfloat16.
- #strong[Статья:] arXiv 2312.06635 --- Gated Linear Attention Transformers with Hardware-Efficient Training
- #strong[Источник конфига:] #link("https://huggingface.co/fla-hub/gla-1.3B-100B/raw/main/config.json")
- #strong[Примечание:] use\_gk = true, use\_gv = false: затухание только по оси ключей. expand\_k = 0.5 вдвое сужает ключи относительно ширины модели --- именно поэтому состояние GLA вдвое меньше, чем у RetNet при том же hidden size.
- #strong[Ссылки на источники:]
  - hidden\_size = 2048, num\_hidden\_layers = 24, num\_heads = 4, expand\_k = 0.5, expand\_v = 1, use\_gk = true, use\_gv = false, use\_output\_gate = true, torch\_dtype = bfloat16 --- fla-hub/gla-1.3B-100B config.json
  - d\_k = 2048 \* 0.5 / 4 = 256, d\_v = 2048 \* 1 / 4 = 512

==== Mamba2
<mamba2>
- #strong[Конфиг:] state-spaces/mamba2-2.7b (опубликованный чекпоинт).
- #strong[`dtype_bytes` = 2] --- не опубликован: в config.json чекпоинта нет поля dtype; для сопоставимости принято 2 байта.
- #strong[Статья:] arXiv 2405.21060 --- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
- #strong[Источник конфига:] #link("https://huggingface.co/state-spaces/mamba2-2.7b/raw/main/config.json")
- #strong[Примечание:] Соответствие обозначениям модуля: H = nheads, d\_k = d\_state, d\_v = headdim. Состояние имеет форму (nheads, headdim, d\_state), а B\_t из SSM играет роль ключа. Обратите внимание на асимметрию: Mamba2 получает число голов за счёт маленького d\_v (64), тогда как семейство дельта-правила выбирает ширину, полагая d\_k = d\_v = 128.
- #strong[Не учтено в столбце состояния:] состояние короткой свёртки, (d\_inner + 2#emph[ngroups];d\_state) \* d\_conv элементов --- ещё 42 KiB (43,008 B) на каждый слой типа «Mamba2 SSD», и это не зависит от `N`. Формула из раздела «Два числа» этого не включает.
- #strong[Ссылки на источники:]
  - d\_model = 2560, n\_layer = 64, ssm\_cfg = {layer: Mamba2} --- state-spaces/mamba2-2.7b config.json
  - d\_state = 128, d\_conv = 4, expand = 2, headdim = 64, ngroups = 1 --- значения по умолчанию Mamba2.#strong[init];, mamba\_ssm/modules/mamba2.py (config.json ни одно из них не переопределяет)
  - d\_inner = expand \* d\_model = 5120; nheads = d\_inner / headdim = 80 --- тот же источник

==== DeltaNet
<deltanet>
- #strong[Конфиг:] fla-hub/delta\_net-1.3B-100B (опубликованный чекпоинт).
- #strong[`dtype_bytes` = 2] --- config.json torch\_dtype = bfloat16.
- #strong[Статья:] arXiv 2406.06484 --- Parallelizing Linear Transformers with the Delta Rule over Sequence Length
- #strong[Источник конфига:] #link("https://huggingface.co/fla-hub/delta_net-1.3B-100B/raw/main/config.json")
- #strong[Примечание:] Самое узкое состояние в таблице при этой ширине: 16 голов по 128x128. beta\_t --- сигмоида (use\_beta = true), строго меньше 1.
- #strong[Не учтено в столбце состояния:] состояние короткой свёртки на q/k/v, (2\*key\_dim + value\_dim) \* conv\_size элементов --- ещё 48 KiB (49,152 B) на каждый слой типа «дельта-правило», и это не зависит от `N`. Формула из раздела «Два числа» этого не включает.
- #strong[Ссылки на источники:]
  - hidden\_size = 2048, num\_hidden\_layers = 24, num\_heads = 16, expand\_k = 1, expand\_v = 1, qk\_norm = \'l2\', qk\_activation = \'silu\', use\_short\_conv = true, conv\_size = 4, use\_beta = true, use\_gate = false, use\_output\_norm = true, torch\_dtype = bfloat16 --- fla-hub/delta\_net-1.3B-100B config.json
  - d\_k = 2048 / 16 = 128, d\_v = 2048 / 16 = 128

==== Gated DeltaNet
<gated-deltanet>
- #strong[Конфиг:] NVlabs/GatedDeltaNet, конфиг `GatedDeltaNet_1.3B` (опубликованный конфиг обучения (официальный релиз кода; чекпоинта на HF нет)).
- #strong[`dtype_bytes` = 2] --- pretrain.py: L.Fabric(..., precision=\'bf16-mixed\').
- #strong[Статья:] arXiv 2412.06464 --- Gated Delta Networks: Improving Mamba2 with Delta Rule
- #strong[Источник конфига:] #link("https://github.com/NVlabs/GatedDeltaNet/blob/b53d6d3a161267432a79c1c04af69fa52bddc921/lit_gpt/config.py")
- #strong[Примечание:] Ловушка, о которой стоит знать: `n_head = 16` в конфиге --- НЕ число голов миксера. lit\_gpt/model.py строит миксер как `GatedDeltaNet(hidden_size=config.n_embd)` и ничего больше, так что действуют собственные значения по умолчанию слоя --- num\_heads = 9, expand\_k = 0.75, expand\_v = 1.5. `n_head` доходит только до слоёв CausalSelfAttention гибридного варианта H1. Поэтому размерности голов получаются 200/400, а не 128, которые рекомендует раздел абляций статьи; ровно на d\_k = 128 попадает конфиг 0.4B (n\_embd = 1536).
- #strong[Не учтено в столбце состояния:] состояние короткой свёртки на q/k/v, (2\*key\_dim + value\_dim) \* conv\_size элементов --- ещё 57,600 B на каждый слой типа «дельта-правило с гейтом», и это не зависит от `N`. Формула из раздела «Два числа» этого не включает.
- #strong[Ссылки на источники:]
  - name = \'GatedDeltaNet\_1.3B\', n\_layer = 16, n\_head = 16, n\_embd = 2400, gated\_delta\_per\_layer = 1 --- NVlabs/GatedDeltaNet lit\_gpt/config.py
  - GatedDeltaNet(hidden\_size=config.n\_embd) со значениями по умолчанию слоя expand\_k = 0.75, expand\_v = 1.5, num\_heads = 9 --- NVlabs/GatedDeltaNet lit\_gpt/model.py, строка 288, и lit\_gpt/gated\_delta\_net.py, строки 34-81
  - key\_dim = 0.75 \* 2400 = 1800 -\> d\_k = 200; value\_dim = 1.5 \* 2400 = 3600 -\> d\_v = 400 --- lit\_gpt/gated\_delta\_net.py, строки 72-81
  - \'a head dimension of 128 provides an optimal trade-off between performance and computational efficiency\' --- arXiv 2412.06464, абляции в приложении

==== Gated DeltaNet-2
<gated-deltanet-2>
- #strong[Конфиг:] NVlabs/GatedDeltaNet-2, конфиг `gdn2_1.3B`; размерности голов сверены с приложением E.1 arXiv 2605.22791 (опубликованный конфиг обучения (официальный релиз кода; чекпоинта на HF нет)).
- #strong[`dtype_bytes` = 2] --- pretrain.py: L.Fabric(..., precision=\'bf16-mixed\'); но см. примечание о типе данных состояния --- само рекуррентное состояние статья хранит в fp32.
- #strong[Статья:] arXiv 2605.22791 --- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention (Ali Hatamizadeh, Yejin Choi, Jan Kautz; submitted 2026-05-21)
- #strong[Источник конфига:] #link("https://github.com/NVlabs/GatedDeltaNet-2/blob/a5552fe3c67e0ebc7ef1220df68ae8896ec62d56/lit_gpt/config.py")
- #strong[Источник реализации:] #link("https://github.com/NVlabs/GatedDeltaNet-2/blob/a5552fe3c67e0ebc7ef1220df68ae8896ec62d56/lit_gpt/gdn2.py")
- #strong[Цитирование перепроверено:] 2026-09-05, через arXiv API (export.arxiv.org/api/query?id\_list=2605.22791), HTML-версию самой статьи и релиз кода на #link("https://github.com/NVlabs/GatedDeltaNet-2") (HTTP 200). Идентификатор, название и авторы совпадают.
- #strong[Примечание:] Смысл этой строки в том, что она --- конечная точка: положив b\_t = beta\_t 1 и w\_t = beta\_t 1, получаем в точности KDA, а если вдобавок схлопнуть alpha\_t до скаляра --- Gated DeltaNet; так что вся последовательность починок из блока 3 умещается в одно уравнение. Статья записывает обновление ровно в обозначениях модуля: S in R^{d\_k x d\_v}, стирающий множитель СЛЕВА, o\_t = S\_t^T q\_t.
- #strong[Источники расходятся, и вот как это разрешено:] Приложение E.1 статьи указывает d\_model = 2048 и нигде не публикует число слоёв. Релиз кода публикует, и он расходится: `gdn2_1.3B` --- это n\_embd = 2304 на 18 слоях. Геометрия голов, задающая размер состояния (H = 16, d\_k = d\_v = 128), в обоих одинакова, так что число на слой не затронуто; число по всей модели использует 18 слоёв из релиза кода --- единственное опубликованное число слоёв.
- #strong[Тип данных состояния:] Приложение D.3: \'The recurrent state is stored in fp32 across chunks and during recurrent decoding.\' При fp32 состояние этой строки на слой --- 1.00 MiB, а не 512 KiB. В таблице эта строка, как и все остальные, посчитана по 2 байта, чтобы сравнивалась ширина; в реальном развёртывании увидят именно цифру для fp32.
- #strong[Не учтено в столбце состояния:] состояние короткой свёртки на q/k/v, (2\*key\_dim + value\_dim) \* conv\_size элементов --- ещё 48 KiB (49,152 B) на каждый слой типа «дельта-правило-2 с гейтом», и это не зависит от `N`. Формула из раздела «Два числа» этого не включает.
- #strong[Ссылки на источники:]
  - name = \'gdn2\_1.3B\' (комментарий: Total parameters 1,302,638,112), n\_layer = 18, n\_head = 18, n\_embd = 2304, gdn2\_per\_layer = 1 (то есть каждый слой --- слой GDN-2) --- NVlabs/GatedDeltaNet-2 lit\_gpt/config.py
  - GatedDeltaNet2(hidden\_size=config.n\_embd) со значениями по умолчанию слоя expand\_v = 1, head\_dim = 128, num\_heads = 16 -\> H = 16, d\_k = d\_v = 128 --- NVlabs/GatedDeltaNet-2 lit\_gpt/model.py, строка 212, и lit\_gpt/gdn2.py, строки 99-133
  - \'Gated DeltaNet, KDA, and Gated DeltaNet-2 use H = 16 heads with d\_k = 128 and d\_v = 128, giving a per-layer recurrent state of H d\_k d\_v = 16 \* 128 \* 128 = 262,144 floats per batch element. Since d\_model = 2048, this equals 128 d\_model.\' --- arXiv 2605.22791, приложение E.1, ур. (90)
  - Ур. (10), Gated Delta Rule-2 в рамке, и ур. (8), (11), (12) для гейтов --- arXiv 2605.22791, раздел 3.1
  - \'The recurrent state is stored in fp32 across chunks and during recurrent decoding.\' --- arXiv 2605.22791, приложение D.3
  - 1.3B параметров, 100B токенов FineWeb-Edu, длина обучения 4K, окно SWA 2K для гибрида --- arXiv 2605.22791, раздел 4 и приложение E.1
  - precision = \'bf16-mixed\' --- NVlabs/GatedDeltaNet-2 pretrain.py, строка 49

==== KDA (Kimi Linear)
<kda-kimi-linear>
- #strong[Конфиг:] moonshotai/Kimi-Linear-48B-A3B-Instruct (48B всего, 3B активных, MoE) (опубликованный чекпоинт).
- #strong[`dtype_bytes` = 2] --- config.json dtype = bfloat16.
- #strong[Статья:] arXiv 2510.26692 --- Kimi Linear: An Expressive, Efficient Attention Architecture
- #strong[Источник конфига:] #link("https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json")
- #strong[Примечание:] Читайте соотношение внимательно. Статья говорит \'a uniform 3:1 ratio\', а выпущенный конфиг --- 20 KDA : 7 MLA на 27 слоях: полное внимание на слоях 4, 8, 12, 16, 20, 24 (это и есть паттерн 3:1 на первых 24 слоях) И на слое 27, последнем. Так что выпущенная модель --- 2.86:1, и её число-прирост на 74.1% ниже полного MLA, а не ровно на 75%. Оба округляются до статейного \'up to 75%\'.
- #strong[Учёт MLA:] Член, линейный по N, здесь --- MLA, который НЕ подчиняется GQA-формуле из раздела «Два числа». MLA кэширует один сжатый латент на токен на слой, шириной kv\_lora\_rank + qk\_rope\_head\_dim = 512 + 64 = 576, общий для всех 32 голов: 1152 байта на токен на слой MLA. Именно на таком учёте держится заявление о 75%. Эталонная HF-реализация (modeling\_kimi.py, KimiMLAAttention.forward) вместо этого материализует в кэш развёрнутые K и V по головам --- 32 \* (192 + 128) = 10,240 элементов, 20,480 байт на токен на слой, \~18x больше. Эффективный \'absorbed\'-путь, который используется в vLLM, кэширует латент. Цифра 75% определяется соотношением слоёв и одинакова при любом из двух учётов.
- #strong[Не учтено в столбце состояния:] состояние короткой свёртки на q/k/v, 3 \* num\_heads \* head\_dim \* kernel элементов --- ещё 96 KiB (98,304 B) на каждый слой типа «KDA (линейное)», и это не зависит от `N`. Формула из раздела «Два числа» этого не включает.
- #strong[Ссылки на источники:]
  - num\_hidden\_layers = 27, linear\_attn\_config.full\_attn\_layers = \[4, 8, 12, 16, 20, 24, 27\], linear\_attn\_config.kda\_layers = 20 элементов, linear\_attn\_config.num\_heads = 32, linear\_attn\_config.head\_dim = 128, linear\_attn\_config.short\_conv\_kernel\_size = 4, kv\_lora\_rank = 512, qk\_rope\_head\_dim = 64, qk\_nope\_head\_dim = 128, v\_head\_dim = 128, num\_attention\_heads = 32, mla\_use\_nope = true, dtype = bfloat16 --- moonshotai/Kimi-Linear-48B-A3B-Instruct config.json
  - \'Kimi Linear interleaves KDA with periodic full attention layers in a uniform 3:1 ratio. This hybrid structure reduces memory and KV-cache usage by up to 75% during long-sequence generation\' --- arXiv 2510.26692, введение
  - \'reducing KV cache usage by up to 75% and achieving up to 6x decoding throughput for a 1M context\' --- arXiv 2510.26692, аннотация
  - \'a uniform 3:1 ratio, i.e., repeating 3 KDA layers to 1 full MLA layer, provided the best quality-throughput trade-off\' --- arXiv 2510.26692, раздел 4, Hybrid model architecture; абляция в таблице 1 (при 3:1 ppl на обучении 9.23 / на валидации 5.65, против 0:1, 1:1, 7:1, 15:1)
  - \'a fixed-sized state (d\_k x d\_v per head, with d\_k = d\_v = 128) regardless of sequence length\' --- arXiv 2510.26692, раздел 6
  - kv\_a\_proj\_with\_mqa проецирует в kv\_lora\_rank + qk\_rope\_head\_dim; past\_key\_values.update кэширует развёрнутые key\_states/value\_states --- modeling\_kimi.py, строки 360-414

==== Qwen3-Next
<qwen3-next>
- #strong[Конфиг:] Qwen/Qwen3-Next-80B-A3B-Instruct (80B всего, 3B активных, MoE) (опубликованный чекпоинт).
- #strong[`dtype_bytes` = 2] --- config.json torch\_dtype = bfloat16.
- #strong[Статья:] Qwen3-Next model card; HF transformers implementation
- #strong[Источник конфига:] #link("https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json")
- #strong[Источник реализации:] #link("https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py")
- #strong[Примечание:] Линейные слои Qwen3-Next --- с точностью до байта эталонный линейный слой лекции: H = 32, d\_k = d\_v = 128, 2 байта --- 1 MiB на слой. Это не совпадение, которое стоит прятать; это самое чистое место во всей таблице, на которое можно указать. Внимание с группировкой значений: linear\_num\_key\_heads = 16, но linear\_num\_value\_heads = 32. q и k размножаются через repeat\_interleave в 2 раза перед рекуррентностью, так что состояние несёт 32 головы, а не 16. Если посчитать здесь 16, строка уменьшится вдвое.
- #strong[Не учтено в столбце состояния:] состояние короткой свёртки, conv\_dim \* linear\_conv\_kernel\_dim элементов --- ещё 64 KiB (65,536 B) на каждый слой типа «Gated DeltaNet (линейное)», и это не зависит от `N`. Формула из раздела «Два числа» этого не включает.
- #strong[Ссылки на источники:]
  - num\_hidden\_layers = 48, full\_attention\_interval = 4, linear\_num\_key\_heads = 16, linear\_num\_value\_heads = 32, linear\_key\_head\_dim = 128, linear\_value\_head\_dim = 128, linear\_conv\_kernel\_dim = 4, num\_attention\_heads = 16, num\_key\_value\_heads = 2, head\_dim = 256, partial\_rotary\_factor = 0.25, torch\_dtype = bfloat16 --- Qwen/Qwen3-Next-80B-A3B-Instruct config.json
  - layer\_types = \[\'linear\_attention\' if bool((i + 1) % interval\_pattern) else \'full\_attention\' for i in range(num\_hidden\_layers)\] при interval\_pattern = full\_attention\_interval = 4, то есть полное внимание при i = 3, 7, ..., 47: 12 из 48 слоёв, ровно 3:1 --- transformers configuration\_qwen3\_next.py, строки 210-217 (коммит 0ed6d51)
  - repeat\_interleave(num\_v\_heads \/\/ num\_k\_heads = 2) для query/key перед chunk\_gated\_delta\_rule, так что рекуррентное состояние имеет 32 головы по 128x128 --- modular\_qwen3\_next.py, строки 597-599 (коммит 0ed6d51)
  - conv\_dim = key\_dim \* 2 + value\_dim = 128#emph[16];2 + 128\*32 = 8192 --- modular\_qwen3\_next.py, строка 445
