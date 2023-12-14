import re
import json
def parse_json(text):
    text = ''.join(c for c in text if c.isprintable())
    text = text.replace('{\n', '{')
    text = text.replace('}\n', '}')
    # text = re.sub(r"'([^\"']+)'", r'"\1"', text) # all pairs as doublequote
    text = re.sub(r"'([^\"']+)':", r'"\1":', text)  # keys as doublequote
    # text = re.sub(r'"([^\'"]+)":', r"'\1':", text) # keys as singlequote
    # text = text.replace("'", '"')
    # text = text.replace("\'", '"')
    start_brace = text.find('{')
    if start_brace >= 0:
        obj_text = text[start_brace:]
        nesting = ['}']
        cleaned = '{'
        in_string = False
        i = 1
        while i < len(obj_text) and len(nesting) > 0:
            ch = obj_text[i]
            if in_string:
                cleaned += ch
                if ch == '\\':
                    i += 1
                    if i < len(obj_text):
                        cleaned += obj_text[i]
                    else:
                        return None
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    nesting.append('}')
                elif ch == '[':
                    nesting.append(']')
                elif ch == '}':
                    close_object = nesting.pop()
                    if close_object != '}':
                        return None
                elif ch == ']':
                    close_array = nesting.pop()
                    if close_array != ']':
                        return None
                elif ch == '<':
                    ch = '"<'
                elif ch == '>':
                    ch = '>"'
                cleaned += ch
            i += 1

        if len(nesting) > 0:
            cleaned += ''.join(reversed(nesting))

        try:
            if type(cleaned) == str:
                obj = json.loads(cleaned)
                return obj
            else:
                return cleaned
            # return obj if len(obj.keys()) > 0 else None
        except json.JSONDecodeError:
            return cleaned
    else:
        return None

def partially_parse_json(text):
    def __find_comma_indices(string):
        indices = []
        for i, char in enumerate(string):
            if char == ',':
                indices.append(i)
        return indices

    indices = __find_comma_indices(text)
    indices.reverse()
    for idx in indices:
        obj_ = parse_json(text[:idx])
        if isinstance(obj_,dict):
            return obj_, text[idx+1:]
    return {}, text
        
def parse_json_wrapper(text):
    text = text.replace("'", "\"")
    clean_output = parse_json(text)
    if isinstance(clean_output, dict):
        return clean_output
    else:
        partial_dict, remain_txt = partially_parse_json(text)
        remain_txt=remain_txt.strip()
        idx_type = 0
        key_ = ""
        val_ = ""
        for char in remain_txt:
            if char =="\"":
                idx_type +=1 
            else:
                if idx_type==1:
                    key_ += char
                if idx_type==3:
                    val_ += char
                continue
            
        partial_dict.update({key_:val_})
        return partial_dict