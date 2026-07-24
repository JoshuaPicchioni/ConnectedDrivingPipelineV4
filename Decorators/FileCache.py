# uses the keyword operator cache_variables to create a long name to store the return of the function in a file
# and read from it if already there
# the function uses the PathProvider but if the key "cache_path" is not found in the context it will use the default
# path of cache/
#
# Task 50: Cache hit rate optimization enhancements
# - Integrated CacheManager for hit/miss tracking (target: >85% hit rate)
# - Improved deterministic cache key generation (sorted cache variables)
# - Added cache hit/miss logging with detailed statistics
# - Cache metadata tracking for LRU eviction support

import functools
from ServiceProviders.PathProvider import PathProvider

import os
import hashlib


def create_deterministic_cache_key(function_name: str, cache_variables: list, context_data=None) -> str:
    """
    Create a deterministic cache key from function name, variables, and full context.

    CRITICAL FIX: Now includes ALL configuration parameters to prevent cache key collisions
    between different configurations. Works with both GeneratorContextProvider instances 
    and context snapshots (dictionaries).

    Args:
        function_name: Name of the cached function
        cache_variables: List of cache variable values
        context_data: GeneratorContextProvider instance, context dict snapshot, or None

    Returns:
        MD5 hash of the deterministic cache key string

    Example:
        >>> create_deterministic_cache_key("process", [100, "data.csv"], {'attack_ratio': 0.3})
        "a3f9e2c1b0d4e5f6g7h8i9j0k1l2m3n4"
    """
    # Start with function name
    key_parts = [function_name]

    # Convert cache variables to strings and handle different types
    str_vars = []
    for var in cache_variables:
        # Handle different types appropriately
        if isinstance(var, (list, tuple)):
            # Recursively handle nested collections
            str_vars.append(str(sorted([str(v) for v in var])))
        elif isinstance(var, dict):
            # Sort dict items for determinism - this is critical for context data
            str_vars.append(str(sorted(var.items())))
        elif hasattr(var, '__dict__') and '<' in str(var) and '>' in str(var):
            # Handle object instances - use class name instead of memory address
            # This prevents cache key changes when object instances change between runs
            str_vars.append(f"{var.__class__.__name__}_{id(var.__class__)}")
        else:
            str_vars.append(str(var))

    # Preserve order for positional arguments - order matters for correctness
    key_parts.extend(str_vars)

    # CRITICAL FIX: Include ALL configuration parameters from context
    # This ensures different configurations (attack ratios, spatial radii, etc.) get unique cache keys
    if context_data is not None:
        try:
            if isinstance(context_data, dict):
                # Context snapshot (dictionary) - already captured
                all_contexts = context_data
            elif hasattr(context_data, 'getAll'):
                # GeneratorContextProvider instance
                all_contexts = context_data.getAll()
            else:
                # Fallback - convert to string
                all_contexts = str(context_data)
            
            if all_contexts and isinstance(all_contexts, dict):
                # Sort context items for determinism but preserve all parameter values
                sorted_contexts = sorted(all_contexts.items())
                key_parts.append(f"CONTEXT_{str(sorted_contexts)}")
            elif all_contexts:
                key_parts.append(f"CONTEXT_{str(all_contexts)}")
        except Exception as e:
            # Fallback - use string representation if processing fails
            key_parts.append(f"CONTEXT_FALLBACK_{str(context_data)}")

    # Join all parts and hash
    key_string = "_".join(key_parts)
    
    # Use UTF-8 encoding for consistency across systems
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()


# decorator to cache the return of a function in a file
# KWARGS:
# cache_variables: list of variables to use as cache variables (default: all the arguments excluding the kwargs)
# cache_file_type: the file type to use for the cache file (default: txt)
# cache_file_reader_function: the function to use to read the file (default: simple read of a txt file)
# cache_file_writer_function: the function to use to write the file (default: simple write of a txt file)
# full_file_cache_path: OVERRIDES the cache path and uses this path instead
# NOTE: the function being decorated must declare the return type in the function declaration

# rewriting as fn decorator
def FileCache(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Import here to avoid circular dependencies
        from Decorators.CacheManager import CacheManager

        KW_ARGS_TO_BE_REMOVED = ["cache_variables", "full_file_cache_path", "cache_file_type", "cache_file_reader_function", "cache_file_writer_function"]

        # Get CacheManager singleton instance for tracking
        cache_manager = CacheManager.get_instance()

        # can be overridden for other file types
        def readFile(file_name, data_type):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "r") as file:
                return data_type(file.read())

        # can be overridden for other file types
        def writeFile(file_path, content):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                file.write(str(content))

        full_path = None

        if not "full_file_cache_path" in kwargs:

            cache_variables = {}
            if "cache_variables" in kwargs:
                cache_variables = kwargs["cache_variables"]
            else:
                # use all the arguments as cache variables
                cache_variables = list(args)

            cache_file_type = "txt"
            # check if "cache_file_type" is in the kwargs
            if "cache_file_type" in kwargs:
                cache_file_type = kwargs["cache_file_type"]

            # check if the file exists with the name of the function and the cache variables

            # filter out the first variable if the str form of the variable has illegal characters for a file name
            # (typically 'self' or object instances with <> in string representation)
            if len(cache_variables) > 0 and "<" in str(cache_variables[0]) and ">" in str(cache_variables[0]):
                cache_variables = cache_variables[1:]

            # CRITICAL FIX: Capture context snapshot at cache key generation time
            # Since GeneratorContextProvider is a singleton, we need to capture 
            # the current state rather than passing the instance
            context_snapshot = None
            if len(args) > 0 and hasattr(args[0], '_generatorContextProvider'):
                try:
                    # Capture a snapshot of current context values
                    context_snapshot = args[0]._generatorContextProvider.getAll().copy()
                except:
                    context_snapshot = None
            elif len(args) > 0 and hasattr(args[0], 'generatorContextProvider'):
                try:
                    # Capture a snapshot of current context values
                    context_snapshot = args[0].generatorContextProvider.getAll().copy()
                except:
                    context_snapshot = None
            
            # Task 50: Use deterministic cache key generation with context snapshot
            file_name = create_deterministic_cache_key(fn.__name__, cache_variables, context_snapshot)

            cache_path = PathProvider().getPathWithModelName("cache_path", lambda name: f"cache/{name}/")
            # create the file path
            full_path = f"{cache_path}{file_name}.{cache_file_type}"

        else:
            full_path = kwargs["full_file_cache_path"]

        # Extract cache key for tracking (file name without extension)
        cache_key = os.path.splitext(os.path.basename(full_path))[0]

        # check if the file exists
        if os.path.exists(full_path):
            # Task 50: Record cache HIT with CacheManager
            cache_manager.record_hit(cache_key, full_path)

            # read the file
            # check if "cache_file_reader_function" is in the kwargs
            if "cache_file_reader_function" in kwargs:
                return kwargs["cache_file_reader_function"](full_path, fn.__annotations__["return"])
            else:
                return readFile(full_path, fn.__annotations__["return"])
        else:
            # Task 50: Record cache MISS with CacheManager
            cache_manager.record_miss(cache_key, full_path)

            # call function and save the return in the file
            return_value = fn(*args, **{k: v for k, v in kwargs.items() if k not in KW_ARGS_TO_BE_REMOVED})
            # check if "cache_file_writer_function" is in the kwargs
            if "cache_file_writer_function" in kwargs:
                kwargs["cache_file_writer_function"](full_path, return_value)
            else:
                writeFile(full_path, return_value)
            return return_value

    return wrapper


@staticmethod
def getCalculatedCachePath(fn, cache_variables, cache_file_type):
    # create the file name
    file_name = fn.__name__
    for cache_variable in cache_variables:
        file_name += "_" + str(cache_variable)

    cache_path = PathProvider().getPathWithModelName("cache_path", lambda name: f"cache/{name}/")
    # create the file path
    full_path = f"{cache_path}{file_name}.{cache_file_type}"

    return full_path


# class FileCache:

#     def __init__(self, fn):
#         self.fn = fn

#         # append the annotations and dict to the function
#         self.fn.__annotations__.update(fn.__annotations__)
#         self.fn.__dict__.update(fn.__dict__)


#     def __call__(self, *args, **kwargs):
#         self.KW_ARGS_TO_BE_REMOVED = ["cache_variables", "full_file_cache_path", "cache_file_type", "cache_file_reader_function", "cache_file_writer_function"]

#         full_path = None

#         if not "full_file_cache_path" in kwargs:

#             cache_variables = {}
#             if "cache_variables" in kwargs:
#                 cache_variables = kwargs["cache_variables"]
#             else:
#                 # use all the arguments as cache variables
#                 cache_variables = list(args)

#             cache_file_type = "txt"
#             # check if "cache_file_type" is in the kwargs
#             if "cache_file_type" in kwargs:
#                 cache_file_type = kwargs["cache_file_type"]

#             # check if the file exists with the name of the function and the cache variables

#             # create the file name
#             file_name = self.fn.__name__
#             for cache_variable in cache_variables:
#                 file_name += "_" + str(cache_variable)

#             cache_path = PathProvider().getPathWithModelName("cache_path", lambda name: f"cache/{name}/")
#             # create the file path
#             full_path = f"{cache_path}{file_name}.{cache_file_type}"

#         else:
#             full_path = kwargs["full_file_cache_path"]

#         # check if the file exists
#         if os.path.exists(full_path):
#             # read the file
#             # check if "cache_file_reader_function" is in the kwargs
#             if "cache_file_reader_function" in kwargs:
#                 return kwargs["cache_file_reader_function"](full_path, self.fn.__annotations__["return"])
#             else:
#                 return self.readFile(full_path, self.fn.__annotations__["return"])
#         else:
#             # call function and save the return in the file
#             return_value = self.fn(self, *args, **{k: v for k, v in kwargs.items() if k not in self.KW_ARGS_TO_BE_REMOVED})
#             # check if "cache_file_writer_function" is in the kwargs
#             if "cache_file_writer_function" in kwargs:
#                 kwargs["cache_file_writer_function"](full_path, return_value)
#             else:
#                 self.writeFile(full_path, return_value)
#             return return_value


#     # can be overridden for other file types
#     def readFile(self, file_name, data_type):
#         os.makedirs(os.path.dirname(file_name), exist_ok=True)
#         with open(file_name, "r") as file:
#             return data_type(file.read())

#     # can be overridden for other file types
#     def writeFile(self, file_path, content):
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, "w") as file:
#             file.write(str(content))

#     @staticmethod
#     def getCalculatedCachePath(fn, cache_variables, cache_file_type):
#         # create the file name
#         file_name = fn.__name__
#         for cache_variable in cache_variables:
#             file_name += "_" + str(cache_variable)

#         cache_path = PathProvider().getPathWithModelName("cache_path", lambda name: f"cache/{name}/")
#         # create the file path
#         full_path = f"{cache_path}{file_name}.{cache_file_type}"

#         return full_path
