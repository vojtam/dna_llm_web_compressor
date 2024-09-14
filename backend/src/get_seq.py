import requests
import re

def get_sequence_ensembl(chromosome, start, end, species="human"):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/region/{species}/{chromosome}:{start}..{end}:1?"
    
    try:
        r = requests.get(server + ext, headers={"Content-Type": "text/plain"})
        r.raise_for_status()
        seq = re.sub('N', '', r.text)
        return seq
    except requests.RequestException as e:
        print(f"Error fetching sequence from Ensembl: {e}")
        return None

# # Usage
# chrom = "8"  # Note: Ensembl uses chromosome numbers without "chr" prefix for human
# start = 127737589
# end = 127737629
# seq = get_sequence_ensembl(chrom, start, end)

# if seq:
#     print(f"Sequence: {seq}")
# else:
#     print("Failed to retrieve sequence")