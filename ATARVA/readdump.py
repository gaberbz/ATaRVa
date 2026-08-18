"""
Optional per-read output for ATaRVa.

Emits one row per (locus, read) giving the read name, which allele cluster the
read was assigned to, and that read's allele length -- i.e. the read-level data
underlying the AL/AR/SD tags in the VCF.

Enabled via the ATARVA_READ_DUMP environment variable, which `atarva genotype
--read-dump` sets. Using an env var rather than threading a new parameter
through cooper/mini_cooper keeps the diff small and rebase-friendly against
upstream; the flag is the user-facing interface.

Each worker process writes its own file alongside its VCF shard, following the
same `_thread_{tidx}` naming that genotype.py already concatenates.
"""

import os

_fh = None
_seen = set()

HEADER = "locus_id\tcontig\tstart\tend\tmotif\tread_name\tallele_idx\tallele_len\tphase_method\n"


def _path(outfile, tidx):
    if tidx == -1 or tidx == 0:
        return f'{outfile}.reads.tsv'
    idx = outfile.rfind('/')
    hid = outfile[:idx + 1] + '.' + outfile[idx + 1:]
    return f'{hid}_thread_{tidx}.reads.tsv'


def init(outfile, tidx):
    """Open this worker's dump file. No-op unless --read-dump was given."""
    global _fh, _seen
    _seen = set()
    if os.environ.get('ATARVA_READ_DUMP') != '1':
        _fh = None
        return
    _fh = open(_path(outfile, tidx), 'w')
    if tidx == -1 or tidx == 0:
        _fh.write(HEADER)


def enabled():
    return _fh is not None


def write(contig, locus_key, global_loci_info, global_loci_variations, hap_read_lists, tag):
    """
    hap_read_lists: list of read-index lists, one per allele cluster.
    Pass [reads] for a homozygous/haploid call, [c1, c2] for a heterozygous one.
    allele_idx in the output is 1-based and matches VCF allele order.
    """
    if _fh is None:
        return
    if locus_key in _seen:          # guard against paths that fall through
        return                       # hetero -> homo within one locus
    row = global_loci_info.get(locus_key)
    if row is None:
        return
    locus_var = global_loci_variations.get(locus_key)
    if locus_var is None:
        return

    _seen.add(locus_key)
    locus_id = row[5] if len(row) > 5 else '.'
    start, end, motif = row[1], row[2], row[3]
    names = locus_var.get('read_name', {})
    seqs = locus_var['read_sequence']
    method = 'SNP' if tag == 'SNP' else 'LENGTH'

    for aidx, hap_reads in enumerate(hap_read_lists, start=1):
        for rid in hap_reads:
            seq = seqs.get(rid)
            if seq is None:
                continue
            _fh.write(f'{locus_id}\t{contig}\t{start}\t{end}\t{motif}\t'
                      f'{names.get(rid, ".")}\t{aidx}\t{len(seq[0])}\t{method}\n')


def close():
    global _fh
    if _fh is not None:
        _fh.close()
        _fh = None
