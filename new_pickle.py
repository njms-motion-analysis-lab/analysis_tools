import dill as pickle
import io

# Define the dummy _unpickle_block function
def _unpickle_block(*args, **kwargs):
    from pandas.core.internals.blocks import Block
    return Block(*args, **kwargs)

# Custom Unpickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == '_unpickle_block':
            return _unpickle_block
        if module == 'pandas._libs.internals' and name == 'Block':
            from pandas.core.internals.blocks import Block
            return Block
        return super().find_class(module, name)

def unpickle_file(data):
    try:
        obj = CustomUnpickler(io.BytesIO(data)).load()
        return obj
    except Exception as e:
        print(f"Error loading pickle data with custom dill unpickler: {e}")
        return None

def pickle_data(obj):
    try:
        return pickle.dumps(obj)
    except Exception as e:
        print(f"Error pickling data with dill: {e}")
        return None