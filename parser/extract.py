import gpmf

def get_gpmf_payloads_from_file(file_path):
    with open(file_path, 'rb') as f:
        payloads, _ = gpmf.extract(f.read())
    return payloads, _
