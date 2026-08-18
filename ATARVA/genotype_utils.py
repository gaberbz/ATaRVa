from ATARVA.snp_utils import haplocluster_reads
from ATARVA import readdump
from ATARVA.vcf_writer import *
from ATARVA.sub_operation_utils import *
from ATARVA.somatic_utils import *

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import stringzilla as sz
   
def homo_vcf_call(alen, read_seqs, haplotypes, DP, amplicon, motif_size, ref, contig, locus_key, global_loci_info, global_loci_variations, out, log_bool, decomp, hallele_counter, tag):

    lower1,upper1 = confidence_interval(alen)
    readdump.write(contig, locus_key, global_loci_info, global_loci_variations, [haplotypes], tag)
    allele_range = f'{lower1}-{upper1},{lower1}-{upper1}'
    ALT, allele_length, decomp_seq, repeativity = alt_sequence(read_seqs, haplotypes, amplicon, motif_size)
    if repeativity:
        meth_info = methylation_calc(haplotypes, global_loci_variations, locus_key, ALT)
        vcf_homozygous_writer(ref, contig, locus_key, global_loci_info, allele_length, len(haplotypes), DP, out, ALT, log_bool, tag, decomp, hallele_counter, False, allele_range, decomp_seq, meth_info)
    else:
        return [False, 6]
    return [True, 10]

def hetero_vcf_call(haplotypes, read_seqs, amplicon, motif_size, new_alen, contig, locus_key, read_indices, global_loci_info, global_loci_variations, locus_start, locus_end, ref, out, log_bool, decomp, hallele_counter, tag):

    alen_c1 = new_alen[0]
    alen_c2 = new_alen[1]
    phased_read = ['.','.']
    chosen_snpQ = '.'
    snp_num = '.'        

    readdump.write(contig, locus_key, global_loci_info, global_loci_variations, haplotypes, tag)

    genotypes = []
    allele_count = []
    ALT_seqs = []
    repeativity_list = []
    decomp_seq_list = []
    meth_info = []
    for hap_reads in haplotypes:
        ALT, allele_length, decomp_seq, repeativity = alt_sequence(read_seqs, hap_reads, amplicon, motif_size)
        repeativity_list.append(repeativity)
        decomp_seq_list.append(decomp_seq)
        ALT_seqs.append(ALT)
        genotypes.append(allele_length)
        allele_count.append(len(hap_reads))

        meth_info.append(methylation_calc(hap_reads, global_loci_variations, locus_key, ALT))

    lower1,upper1 = confidence_interval(alen_c1)
    lower2,upper2 = confidence_interval(alen_c2)
    allele_range = f'{lower1}-{upper1},{lower2}-{upper2}'

    if all(repeativity_list):
        vcf_heterozygous_writer(contig, genotypes, locus_start, locus_end, allele_count, len(read_indices), global_loci_info, ref, out, chosen_snpQ, phased_read, snp_num, ALT_seqs, log_bool, tag, decomp, hallele_counter, allele_range, decomp_seq_list, meth_info)
    elif any(repeativity_list):
        if repeativity_list[0]:
            no_intracluster = True # will be always true in WGS default mode, to run the homozygous part of the code. This is avoid extra if,else conditon if dbscan fails inside amplicon
            if amplicon:
                db_status, new_hap, new_alen = dbscan(alen_c1, haplotypes[0])
                if db_status:
                    no_intracluster = False
                    bool_state, category = hetero_vcf_call(new_hap, read_seqs, amplicon, motif_size, new_alen, contig, locus_key, read_indices, global_loci_info, global_loci_variations, locus_start, locus_end, ref, out, log_bool, decomp, hallele_counter, tag)
                    return [bool_state, category]
            if no_intracluster: # if amplicon is true but there are no intraclusters, then give it as homozygous
                allele_range = f'{lower1}-{upper1},{lower1}-{upper1}'
                vcf_homozygous_writer(ref, contig, locus_key, global_loci_info, genotypes[0], len(haplotypes[0]), len(read_indices), out, ALT_seqs[0], log_bool, tag, decomp, hallele_counter, False, allele_range, decomp_seq_list[0], meth_info[0])
        else:
            no_intracluster = True # will be always true in WGS default mode, to run the homozygous part of the code
            if amplicon:
                db_status, new_hap, new_alen = dbscan(alen_c2, haplotypes[1])
                if db_status:
                    no_intracluster = False
                    bool_state, category = hetero_vcf_call(new_hap, read_seqs, amplicon, motif_size, new_alen, contig, locus_key, read_indices, global_loci_info, global_loci_variations, locus_start, locus_end, ref, out, log_bool, decomp, hallele_counter, tag)
                    return [bool_state, category]
            if no_intracluster: # if amplicon is true but there are no intraclusters, then give it as homozygous
                allele_range = f'{lower2}-{upper2},{lower2}-{upper2}'
                vcf_homozygous_writer(ref, contig, locus_key, global_loci_info, genotypes[1], len(haplotypes[1]), len(read_indices), out, ALT_seqs[1], log_bool, tag, decomp, hallele_counter, False, allele_range, decomp_seq_list[1], meth_info[1])
    else:
        return [False, 6]
    
    return [True, 10]


def score_calc(x_grid, density, initial_peaks, valleys, top_contour_widths):
    peak_density = density[initial_peaks]
    valley_density = density[valleys]
    peak_points = x_grid[initial_peaks]
    initial_score = []
    initial_prominence = []
    initial_skewness = []
    area_covered = []
    for idx in range(len(peak_density)):
        if idx == 0:
            f_dense = density[0]
            base_left = x_grid[0][0]
        if idx < len(peak_density)-1:
            valley_dense = valley_density[idx]
            base_right = x_grid[valleys[idx]][0]
        else:
            valley_dense = density[-1]
            base_right = x_grid[-1][0]

        max_point = max(f_dense, valley_dense)
        prominence = (peak_density[idx] - max_point)# - diffs
        f_dense = valley_dense

        flattened_x_grid = x_grid[:, 0]
        mask = (flattened_x_grid >= base_left) & (flattened_x_grid <= base_right)
        x_vals = flattened_x_grid[mask]
        y_vals = density[mask]

        area = np.trapz(y_vals, x_vals)
        area_covered.append(area)
        min_area = 1 if area >= 0.02 else 0 # min area covered by the peak should be atleast 2% to be considered as valid peak

        current_peak = peak_points[idx][0]
        L = (current_peak - base_left); R = (base_right-current_peak) # distance between left boundary to the peak and right boundary to the peak
        
        eps = 1e-8
        # normalized asymmetry/skewness
        K = abs(L - R) / (L + R + eps)
        initial_skewness.append(K)
        
        # final score
        score = (
            (prominence ** 2) / ((top_contour_widths[idx] + eps) ** 2)
        ) * min_area


        initial_score.append(score)
        initial_prominence.append(prominence)
        
        base_left = base_right

    initial_skewness = np.array(initial_skewness)
    median_K = np.median(initial_skewness)
    mad_K = np.median( np.abs( initial_skewness - median_K ) ) + 1e-8
    zK = (initial_skewness - median_K) / mad_K
    quality_zk = 1 / (1 + np.exp(zK))

    final_score = ( np.array(initial_score) * quality_zk )
        
    return final_score, initial_prominence, area_covered


def length_genotyper(hallele_counter, global_loci_info, global_loci_variations, locus_key, read_indices, contig, locus_start, locus_end, ref, out, male, log_bool, decomp, read_seqs, amplicon):

    read_indices = sorted(read_indices)
    locus_read_allele = global_loci_variations[locus_key]['read_allele']
    unique_alen = list(hallele_counter.keys())
    motif_size = int(float(global_loci_info[locus_key][4])) # <= 10 # boolean for motif-decomp check
    

    alen_with_1read = [item[0] for item in hallele_counter.items() if item[1]==1] # allele with 1 read contribution

    main_read_id = []
    alen_data = []
    
    for id in read_indices:
        if locus_read_allele[id][0] in alen_with_1read: # checking if the '1 read - allele' is nearby any of other 'good read - allele'
            num = locus_read_allele[id][0]
            for i in set(unique_alen): #for i in alen_with_gread:
                if i == num: continue
                window = round(0.1*i)
                if (i-window) <= num <= (i+window): # '1 read - allele' is considered if other allele are within 10% on either of the side
                    alen_data.append(num)
                    main_read_id.append(id)
                    break
        else:
            alen_data.append(locus_read_allele[id][0])
            main_read_id.append(id)

    if len(alen_data) < 3:
        return [False, 6]

    if amplicon:
        tag = 'KDE'
        data = np.array([length//motif_size for length in alen_data])
        #### KDE with mode peaks and valley for definitive split point for each peak based on the area under the peaks
        bandwidth = 10; tot_data_points = 1000 # for amplicon, to get better density estimation and peaks
        stdev = np.std(data)
        if stdev != 0:
            bandwidth = 0.5 * stdev * (len(data) ** (-1/5))

        data = data.reshape(-1, 1)

        # Fit kde to the data
        kde = KernelDensity(kernel='gaussian', algorithm='kd_tree', metric='minkowski', bandwidth=bandwidth).fit(data)
        # Evaluate the density on a grid
        x_grid = np.linspace(data.min()-50, data.max()+50, tot_data_points).reshape(-1, 1)
        log_density = kde.score_samples(x_grid)
        density = np.exp(log_density)

        # Analysing the distribution to identify the sharp narrow peaks
        initial_peaks, _ = find_peaks(density)
        original_widths = peak_widths(density, initial_peaks)
        top_contour_widths = peak_widths(density, initial_peaks, rel_height=0.2)
        valleys, _ = find_peaks(-density)

        score, initial_prominence, area_covered = score_calc(x_grid, density, initial_peaks, valleys, top_contour_widths[0])
        narrow_peaks_idx = list(np.argsort(score)[-2:]) # taking top two peaks with more area under the curve, as the peaks with higher area will be sharper and more prominent

        top_2_prominence = [initial_prominence[idx] for idx in narrow_peaks_idx]
        min_height_covered = min(top_2_prominence) >= 0.15 * (max(top_2_prominence)) # min peak should have atleast 15% of the max peak prominence to be considered as a valid peak
        min_area_covered = all([area_covered[idx]>=0.05 for idx in narrow_peaks_idx]) # both peaks should have atleast 5% of the area covered to be considered as valid peaks

        if min_height_covered or min_area_covered: # any of this should be True to consider both peaks as valid peaks, otherwise only the max peak will be considered for split
            pass
        elif (narrow_peaks_idx[0] > narrow_peaks_idx[1]) and (area_covered[narrow_peaks_idx[0]] >= 0.02): # if the min_peak is on right side of the max_peak and if it has atleast 2% of total area consider both peaks as valid; to report the longer allele
            pass
        else: # if the min_peak is left side of the max_peak, then consider only the max_peak as valid and as homozygous 
            narrow_peaks_idx = [narrow_peaks_idx[1]]

        width = sorted(original_widths[0][narrow_peaks_idx])
        top_count = len(narrow_peaks_idx)

        # Getting new peaks with analysed data
        peaks, _ = find_peaks(density, width = width)

        # Choose split 
        peak_heights = density[peaks] # extracting only the peaks frim density
        top_peaks = peaks[np.argsort(peak_heights)[-top_count:]] # taking top two peaks
        sorted_peaks = sorted(top_peaks)

        # flattening the data and initializing the labels for each data point as -1 (unassigned)
        data = data.flatten()
        labels = np.full(len(data), -1)

        left = sorted_peaks[0]
        # boundaries of left peak
        peak1_left = valleys[(valleys < left)]
        p1_start = peak1_left[-1] if peak1_left.size > 0 else 0
        peak1_right = valleys[(valleys > left)]
        p1_end = peak1_right[0] if peak1_right.size > 0 else len(x_grid) - 1

        p1_left_split = x_grid[p1_start][0]
        p1_right_split = x_grid[p1_end][0]

        p1_allele_bool = (data >= p1_left_split) & (data <= p1_right_split)
        if p1_allele_bool.sum() > 4: # atleast 5 reads should be there in cluster to consider it as valid cluster
            labels[p1_allele_bool] = 0

        if len(sorted_peaks) > 1:
            right = sorted_peaks[1]
            # boundaries of right peak
            peak2_left = valleys[(valleys < right)]
            p2_start = peak2_left[-1] if peak2_left.size > 0 else 0
            peak2_right = valleys[(valleys > right)]
            p2_end = peak2_right[0] if peak2_right.size > 0 else len(x_grid) - 1

            p2_left_split = x_grid[p2_start][0]
            p2_right_split = x_grid[p2_end][0]

            labels[(data >= p2_left_split) & (data <= p2_right_split)] = 1 # no min read cutoff for the longer allele

            # p2_allele_bool = (data >= p2_left_split) & (data <= p2_right_split)
            # if p2_allele_bool.sum() > 4: # atleast 5 reads should be there in cluster to consider it as valid cluster
            #     labels[(data >= p2_left_split) & (data <= p2_right_split)] = 1

    else:
        tag = 'Ed_Hdbscan'
        data = np.array(alen_data)
        data = data.reshape(-1, 1)

        ref_seq = ref.fetch(contig, locus_start, locus_end)
        edit_list = []
        for read_id in main_read_id:
            current_seq = read_seqs[read_id][0]
            current_seq_len = len(current_seq)

            edit_list.append( [sz.edit_distance(ref_seq, current_seq),  current_seq_len] ) 

        del current_seq
        db_status, labels, _ = dbscan(edit_list, None, 0.1)

        if not db_status: # if dbscan fails, assign it as homozygous
            labels = [0]*len(main_read_id)

    c1 = [i for i, x in enumerate(labels) if x == 0]
    c2 = [i for i, x in enumerate(labels) if x == 1]

    alen_c1 = [alen_data[i] for i in c1]
    alen_c2 = [alen_data[i] for i in c2]

    haplotypes = ([main_read_id[idx] for idx in c1], [main_read_id[idx] for idx in c2])
    cutoff = 0.15*len(alen_data) # 15%

    br = False
    if c1 and c2:
        def process_conditions(alen_x, alen_y):
            nonlocal br, cutoff
            min_val, max_val = confidence_interval(alen_y)
            min_slide = 10 if min_val*0.1 > 10 else min_val*0.1  # max(min_val*0.1, 10)#; maximum length slide is 10
            max_slide = 10 if max_val*0.1 > 10 else max_val*0.1  # max(max_val*0.1, 10)#; maximum length slide is 10
            min_bound = min_val-min_slide
            max_bound = max_val+max_slide

            for min_al in alen_x:
                if min_bound <= min_al <= max_bound:
                    br = True
                    break

            if not br:
                cutoff = int(max(0.03, len(alen_x) / len(alen_data)) * len(alen_data)) # min 3 % of total reads should be in the cluster
                cutoff = max(2, cutoff) # min 2 reads should be there in cluster if WGS
                if amplicon:
                    cutoff = min(5, cutoff)

        if len(c1) < cutoff and len(c2) >= cutoff:
            process_conditions(alen_c1, alen_c2)
                               
        elif len(c2) < cutoff and len(c1) >= cutoff:
            process_conditions(alen_c2, alen_c1)

    cutoff = int(cutoff)

    if male:
        cluster_len = [len(c1), len(c2)]
        cidx = cluster_len.index(max( cluster_len ))
        if cluster_len[cidx]>=cutoff:
            mac = haplotypes[cidx]
            mal = alen_c1 if cidx==0 else alen_c2 # major allele length cluster
            lower,upper = confidence_interval(mal)
            allele_range = f'{lower}-{upper}'
            ALT, allele_length, decomp_seq, repeativity = alt_sequence(read_seqs, mac, amplicon, motif_size)
            meth_info = methylation_calc(mac, global_loci_variations, locus_key, ALT)
            if repeativity:
                vcf_homozygous_writer(ref, contig, locus_key, global_loci_info, allele_length, len(mac), len(read_indices), out, ALT, log_bool, tag, decomp, hallele_counter, True, allele_range, decomp_seq, meth_info)
            else:
                return [False, 6]
        else:
            return [False, 6]
        
    elif (c1!=[] and len(c1)>=cutoff) and (c2!=[] and len(c2)>=cutoff):

        bool_state, category = hetero_vcf_call(haplotypes, read_seqs, amplicon, motif_size, [alen_c1, alen_c2], contig, locus_key, read_indices, global_loci_info, global_loci_variations, locus_start, locus_end, ref, out, log_bool, decomp, hallele_counter, tag)
        return [bool_state, category]
    
    elif c1!=[] and len(c1)>=cutoff:
        if amplicon:
            db_status, new_hap, new_alen = dbscan(alen_c1, haplotypes[0])
            if db_status:
                bool_state, category = hetero_vcf_call(new_hap, read_seqs, amplicon, motif_size, new_alen, contig, locus_key, read_indices, global_loci_info, global_loci_variations, locus_start, locus_end, ref, out, log_bool, decomp, hallele_counter, tag)
                return [bool_state, category]
            else:
                bool_state, category = homo_vcf_call(alen_c1, read_seqs, haplotypes[0], len(read_indices), amplicon, motif_size, ref, contig, locus_key, global_loci_info, global_loci_variations, out, log_bool, decomp, hallele_counter, tag)
                return [bool_state, category]
        else:
            bool_state, category = homo_vcf_call(alen_c1, read_seqs, haplotypes[0], len(read_indices), amplicon, motif_size, ref, contig, locus_key, global_loci_info, global_loci_variations, out, log_bool, decomp, hallele_counter, tag)
            return [bool_state, category]
        
    elif c2!=[] and len(c2)>=cutoff:
        if amplicon:
            db_status, new_hap, new_alen = dbscan(alen_c2, haplotypes[1])
            if db_status:
                bool_state, category = hetero_vcf_call(new_hap, read_seqs, amplicon, motif_size, new_alen, contig, locus_key, read_indices, global_loci_info, global_loci_variations, locus_start, locus_end, ref, out, log_bool, decomp, hallele_counter, tag)
                return [bool_state, category]
            else:
                bool_state, category = homo_vcf_call(alen_c2, read_seqs, haplotypes[1], len(read_indices), amplicon, motif_size, ref, contig, locus_key, global_loci_info, global_loci_variations, out, log_bool, decomp, hallele_counter, tag)
                return [bool_state, category]
        else:
            bool_state, category = homo_vcf_call(alen_c2, read_seqs, haplotypes[1], len(read_indices), amplicon, motif_size, ref, contig, locus_key, global_loci_info, global_loci_variations, out, log_bool, decomp, hallele_counter, tag)
            return [bool_state, category]
        
    else:
        return [False, 6] # write allele distribution with only one read supporting to it in vcf
    
    return [True, 10]


def analyse_genotype(contig, locus_key, global_loci_info,
                     global_loci_variations, global_read_variations, global_snp_positions, hallele_counter,
                     ref, out, sorted_global_snp_list, snpQ, snpC, snpD, snpR, phasingR, read_indices, male, log_bool, decomp, amplicon, somatic):
            
    locus_start = int(global_loci_info[locus_key][1])
    locus_end = int(global_loci_info[locus_key][2])
    motif_size = int(float(global_loci_info[locus_key][4]))

    state = False

    read_seqs = global_loci_variations[locus_key]['read_sequence']

    if somatic: # for somatic variant calling
        state, skip_point, genotype_dict = correlation_clustering(read_seqs, read_indices, motif_size, global_loci_variations, locus_key)
        if state:
            vcf_multizygous_writer(contig, genotype_dict, locus_start, locus_end, len(read_indices), global_loci_info, ref, out, log_bool, decomp, hallele_counter)
        return [state, skip_point]
    
    # elif male or amplicon: # for haploid and amplicon genotyping # -------------- commented this to do SNP clustering for both WGS & AMPLICON ------------------
    elif male: # for haploid genotyping
        state, skip_point = length_genotyper(hallele_counter, global_loci_info, global_loci_variations, locus_key, read_indices, contig, locus_start, locus_end, ref, out, male, log_bool, decomp, read_seqs, amplicon)
        return [state, skip_point]


    snp_positions = set()
    for rindex in read_indices:
        snp_positions |= (global_read_variations[rindex]['snps'])

    snp_positions = sorted(list(filter(lambda x: (x in global_snp_positions) and (global_snp_positions[x]['cov'] >= 3) and
                                                    (locus_start - snpD < x < locus_end + snpD),
                            snp_positions)))

    snp_allelereads = {}
    read_indices = set(read_indices)
    non_ref_snp_cov = {}
    for pos in snp_positions:
        c_point=0
        coverage = set()
        non_ref_nucs = [nucleotides for nucleotides in global_snp_positions[pos] if nucleotides not in ['cov', 'Qval', 'r']]
        for each_nuc in non_ref_nucs:
            reads_of_nuc = global_snp_positions[pos][each_nuc].intersection(read_indices)
            if len(reads_of_nuc) == 0: continue
            coverage.add(len(reads_of_nuc))

            if (sum([global_snp_positions[pos]['Qval'][read_idx] for read_idx in reads_of_nuc])/len(reads_of_nuc)) <= snpQ:
                c_point=1
                break
        if (len(coverage)==0) or (c_point==1): continue
        else: non_ref_snp_cov[pos] = max(coverage)
            
        snp_allelereads[pos] = { 'cov': 0, 'reads': set(), 'alleles': {}, 'Qval': {} }
        for nuc in global_snp_positions[pos]:
            if (nuc == 'cov') or (nuc == 'Qval'): continue
            snp_allelereads[pos]['alleles'][nuc] = global_snp_positions[pos][nuc].intersection(read_indices)
            snp_allelereads[pos]['cov'] += len(snp_allelereads[pos]['alleles'][nuc])
            if nuc!='r':
                snp_allelereads[pos]['Qval'].update(dict([(read_idx,global_snp_positions[pos]['Qval'][read_idx]) for read_idx in snp_allelereads[pos]['alleles'][nuc]]))

    del_positions = list(filter(lambda x: snp_allelereads[x]['cov'] < 5, snp_allelereads.keys()))

    for pos in del_positions:
        del snp_allelereads[pos]


    ordered_snp_on_cov = sorted(snp_allelereads.keys(), key = lambda item : non_ref_snp_cov[item], reverse = True)

    haplotypes, min_snp, skip_point, chosen_snpQ, phased_read, snp_num = haplocluster_reads(snp_allelereads, ordered_snp_on_cov, read_indices, snpC, snpR, phasingR) # SNP ifo and supporting reads for specific locus are given to the phasing function

    if haplotypes == (): # if the loci has no significant snps
        state, skip_point = length_genotyper(hallele_counter, global_loci_info, global_loci_variations, locus_key, read_indices, contig, locus_start, locus_end, ref, out, male, log_bool, decomp, read_seqs, amplicon)
        del read_seqs
        return [state, skip_point]
    
    if min_snp != -1:
        snp_left_boundary = locus_start - snpD
        min_idx = 0
        for each_spos in sorted_global_snp_list:
            if each_spos >= snp_left_boundary:
                break
            del global_snp_positions[each_spos]
            min_idx += 1
        del sorted_global_snp_list[:min_idx]


    readdump.write(contig, locus_key, global_loci_info, global_loci_variations, haplotypes, 'SNP')

    genotypes = []
    allele_count = []
    ALT_seqs = []
    alen_list = []
    meth_info = []
    for hap_reads in haplotypes:
        ALT, allele_length,_,_ = alt_sequence(read_seqs, hap_reads, False, motif_size)
        alen_list.append([len(read_seqs[read_id][0]) for read_id in hap_reads])
        ALT_seqs.append(ALT)
        genotypes.append(allele_length)
        allele_count.append(len(hap_reads))
        # if allele_length not in allele_count:
        #     allele_count[allele_length] = len(hap_reads)
        # else:
        #     allele_count[str(allele_length)] = len(hap_reads)

        meth_info.append(methylation_calc(hap_reads, global_loci_variations, locus_key, ALT))

    del read_seqs
    lower1,upper1 = confidence_interval(alen_list[0])
    lower2,upper2 = confidence_interval(alen_list[1])
    allele_range = f'{lower1}-{upper1},{lower2}-{upper2}'
    vcf_heterozygous_writer(contig, genotypes, locus_start, locus_end, allele_count, len(read_indices), global_loci_info, ref, out, chosen_snpQ, phased_read, snp_num, ALT_seqs, log_bool, 'SNP', decomp, hallele_counter, allele_range, [None], meth_info)
    state = True
    return [state, skip_point]
    
