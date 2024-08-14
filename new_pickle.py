import dill as pickle
import io
from pandas.core.internals.blocks import Block as PandasBlock
from pandas._libs.internals import BlockPlacement

def _unpickle_block(*args, **kwargs):
    if len(args) > 0 and isinstance(args[1], slice):
        args = (args[0], BlockPlacement(range(*args[1].indices(args[1].stop))), *args[2:])
    if 'placement' in kwargs and isinstance(kwargs['placement'], slice):
        kwargs['placement'] = BlockPlacement(range(*kwargs['placement'].indices(kwargs['placement'].stop)))
    return PandasBlock(*args, **kwargs)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(f"NAME, MODULE {name} {module}")
        if name == '_unpickle_block':
            return _unpickle_block
        if module == 'pandas._libs.internals' and name == 'Block':
            return PandasBlock
        if module == 'pandas.core.indexes.numeric':
            import pandas as pd
            if hasattr(pd.core.indexes, 'numeric'):
                return getattr(pd.core.indexes.numeric, name)
            if hasattr(pd.core.indexes.api, name):
                return getattr(pd.core.indexes.api, name)
            if hasattr(pd.core.indexes, name):
                return getattr(pd.core.indexes, name)
        if module == 'pandas._libs.internals' and name == 'BlockPlacement':
            return BlockPlacement
        return super().find_class(module, name)

def unpickle_file(data):
    try:
        obj = CustomUnpickler(io.BytesIO(data)).load()
        return obj
    except Exception as e:
        print(f"Error loading pickle data with custom dill unpickler: {e}")
        import traceback
        traceback.print_exc()
        return None

def unpickle_file(data):
    try:
        obj = CustomUnpickler(io.BytesIO(data)).load()
        return obj
    except Exception as e:
        print(f"Error loading pickle data with custom dill unpickler: {e}")
        import traceback
        traceback.print_exc()
        return None

def pickle_data(obj):
    try:
        return pickle.dumps(obj)
    except Exception as e:
        print(f"Error pickling data with dill: {e}")
        return None

def inspect_pickle(data):
    buffer = io.BytesIO(data)
    try:
        while True:
            try:
                opcode = buffer.read(1)
                if not opcode:
                    break
                op = pickle.opmap.get(ord(opcode))
                if op:
                    print(f"Opcode: {op}")
                    if op == 'GLOBAL':
                        module = buffer.readline().decode('utf-8').strip()
                        name = buffer.readline().decode('utf-8').strip()
                        print(f"  Module: {module}, Name: {name}")
            except Exception as e:
                print(f"Error during inspection: {e}")
                break
    except EOFError:
        pass
    print("Pickle inspection complete")