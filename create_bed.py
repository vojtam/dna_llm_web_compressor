import random

# Chromosome 8 length for GRCh38 (from UCSC Genome Browser)
chr8_length = 145138636  # in base pairs

# Parameters
num_intervals = 10000  # Number of intervals to cover chromosome 8

# Generate random intervals across the chromosome
interval_size = chr8_length // num_intervals

# Create a BED file content
bed_content = []

start = 0
for i in range(num_intervals):
    end = min(start + interval_size, chr8_length)  # Ensure we don't exceed chr8 length
    value = random.randint(1, 1000)  # Random integer as a placeholder for the "name"
    bed_content.append(f"chr8\t{start}\t{end}\t{value}")
    start = end

# Write to a BED file
with open("chr8_grch38.bed", "w") as bed_file:
    bed_file.write("\n".join(bed_content))

print("BED file 'chr8_grch38.bed' created successfully.")