def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v:k for k,v in key_val.items()}

# Mutates the dictionary!!
def replace_key(my_dict: dict, old_key, new_key) -> dict:
    my_dict[new_key] = my_dict.pop(old_key)
    return my_dict
