from Bio import SeqIO
import gzip

def quick_sample_check(file_path, sample_size=10000):
    """Check a random sample without full processing."""
    lengths = []

    handle = gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')

    with handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            lengths.append(len(record.seq))
            if i >= sample_size:
                break

    print(f"Sample of first {len(lengths)} sequences:")
    print(f"  30-1024 aa: {sum(1 for l in lengths if 30 <= l <= 1024)} ({100*sum(1 for l in lengths if 30 <= l <= 1024)/len(lengths):.1f}%)")
    print(f"  < 30 aa: {sum(1 for l in lengths if l < 30)} ({100*sum(1 for l in lengths if l < 30)/len(lengths):.1f}%)")
    print(f"  > 1024 aa: {sum(1 for l in lengths if l > 1024)} ({100*sum(1 for l in lengths if l > 1024)/len(lengths):.1f}%)")

    return lengths

# Run this first
lengths = quick_sample_check("./data/uniref50.fasta.gz", sample_size=100000)
print(lengths)