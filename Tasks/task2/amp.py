import torch
import torch.nn as nn
import torch.nn.functional as F


class Autocast:

    def __init__(self, enabled=True, dtype=torch.float16):
        self.enabled = enabled and torch.cuda.is_available()
        self.target_dtype = dtype
        self._original_funcs = {}
        self._depth = 0

        # Define the functions we want to intercept and cast.
        # Format: (module, function_name_string)
        self.DOWNCAST_OPS = [
            (torch, 'matmul'),
            (torch, 'bmm'),
            (F, 'linear'),
        ]
        self.UPCAST_OPS = [
            (F, 'layer_norm'),
            (F, 'cross_entropy'),
        ]


    def _create_wrapper(self, original_func, target_dtype):
        """
        Creates a wrapper that casts inputs to the target dtype
        and calls the function with casted arguments
        """
        def wrapper(*args, **kwargs):
            """
            Wrapper that casts all inputs to the target dtype 
            and calls the function with casted arguments
            """

            # Cast all tensor arguments in args and kwargs
            ### YOUR CODE HERE

            # Call the original function with potentially casted inputs
            ### YOUR CODE HERE
            pass

        return wrapper

    def __enter__(self):
        """
        Wraps all the functions from DOWNCAST_OPS and UPCAST_OPS,
        Stores original functions
        And sets wrapped functions instead of original ones in the module
        """
        if not self.enabled:
            return self

        self._depth += 1
        if self._depth > 1:
            return self

        # Store original functions and apply patches
        for module, func_name in self.DOWNCAST_OPS + self.UPCAST_OPS:
            # Store original function
            ### YOUR CODE HERE

            # Create wrapped version of the function
            # Note that you need different target_dtype for DOWNCAST_OPS and UPCAST_OPS
            ### YOUR CODE HERE

            # Set wrapped function as attribute of the module with the same name as original function
            ### YOUR CODE HERE
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restores original function
        """
        if not self.enabled:
            return

        self._depth -= 1
        if self._depth > 0:
            return False

        # Restore original functions
        ### YOUR CODE HERE
        
        # Clear the stored functions for the next use
        ### YOUR CODE HERE
        return False


class StaticGradScaler:
    def __init__(self, scale):
        """
        scale: loss scaling coef
        """
        pass
    
    def scale(self, loss):
        """Scales the loss"""
        ### YOUR CODE HERE
        pass
    
    def step(self, optimizer):
        """
        Performs single optimization step
        """
        # Ignore parameters whose grad is None. Check every gradient before
        # unscaling any of them, then unscale without using .data.
        # Unscale the gradients
        # Perform optimizer step
        # Skip optimizer step if there is any nan/inf in the gradient
        # Do not forget that torch accumulates gradients
        ### YOUR CODE HERE
        pass
    
    def update(self):
        """Updates scaling coef"""
        pass


class DynamicGradScaler:
    def __init__(self, scale, factor, patience, min_scale, max_scale):
        """
        scale: initial value of loss scaling coef
        factor: multiplier that used for scaling coef increase/decrease
        patience: how many iters there should be no nan/inf to increase scale
        min_scale: minimal allowed scaling coef value
        max_scale: maximal allowed scaling coef value
        """
        ### YOUR CODE HERE
        pass

    def scale(self, loss):
        """Scales the loss"""
        ### YOUR CODE HERE
        pass

    def step(self, optimizer):
        """
        Performs single optimization step
        """
        # Ignore parameters whose grad is None. Check every gradient before
        # unscaling any of them, then unscale without using .data.
        # Unscale the gradients
        # If there is any nan/inf in the gradient decrease scaling coef using factor
        # Note that scaling coef should be greater than min_scale
        # Perform optimizer step
        # Skip optimizer step if there is any nan/inf in the gradient
        # Do not forget that torch accumulates gradients
        ### YOUR CODE HERE
        pass


    def update(self):
        """Updates scaling coef"""
        # If there was no any nan/inf in the gradient patience steps, increase scaling coef using factor
        # Note that scaling coef should be smaller than max_scale
        ### YOUR CODE HERE
        pass
