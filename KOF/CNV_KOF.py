import numpy as np
import pysam
import os
import gc
import pandas as pd
from numba import njit
import rpy2.robjects as robjects
import datetime
import math
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances


def read_bam(file):
    # reading bam file
    sam_file = pysam.AlignmentFile(file, "rb")
    chr_list = sam_file.references
    return chr_list


def read_ref(file, chr_num, ref):
    # reading reference file
    if os.path.exists(file):
        print("Reading reference file: " + str(file))
        with open(file, 'r') as f:
            f.readline()
            for line in f:
                lines = line.strip()
                ref[chr_num] += lines
    else:
        print("Warning: Cannot open " + str(file) + '\n')
    return ref


def bins(ref, bin_size, chr_len, file):
    chr_tag = np.full(23, 0)
    chr_list = np.arange(23)
    max_num = int(chr_len.max() / bin_size) + 1
    init_rd = np.full((23, max_num), 0.0)
    # read bam file and get bin rd
    print("Read bam file: " + str(file))
    sam_file = pysam.AlignmentFile(file, "rb")
    for line in sam_file:
        idx = int(line.pos / bin_size)
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                init_rd[int(chr)][idx] += 1
                chr_tag[int(chr)] = 1
    chr_list = chr_list[chr_tag > 0]
    chr_num = len(chr_list)
    rd_list = [[] for _ in range(chr_num)]
    pos_list = [[] for _ in range(chr_num)]
    init_gc = np.full((chr_num, max_num), 0)
    pos = np.full((chr_num, max_num), 0)
    # initialize bin_data and bin_head
    count = 0
    for i in range(len(chr_list)):
        chr = chr_list[i]
        bin_num = int(chr_len[chr] / bin_size) + 1
        for j in range(bin_num):
            pos[i][j] = j
            cur_ref = ref[chr][j * bin_size:(j + 1) * bin_size]
            N_count = cur_ref.count('N') + cur_ref.count('n')
            if N_count == 0:
                gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
            else:
                gc_count = 0
                init_rd[chr][j] = -1000000
                count = count + 1
            init_gc[i][j] = int(round(gc_count / bin_size, 3) * 1000)
        # delete
        cur_rd = init_rd[chr][:bin_num]
        cur_gc = init_gc[i][:bin_num]
        cur_pos = pos[i][:bin_num]
        cur_rd = cur_rd / 1000
        index = cur_rd >= 0
        rd = cur_rd[index]
        GC = cur_gc[index]
        cur_pos = cur_pos[index]
        rd[rd == 0] = mode_rd(rd)
        rd = gc_correct(rd, GC)
        pos_list[i].append(cur_pos)
        rd_list[i].append(rd)
    del init_rd, init_gc, pos
    gc.collect()
    return rd_list, pos_list, chr_list


def mode_rd(rd):
    new_rd = np.full(len(rd), 0)
    for i in range(len(rd)):
        new_rd[i] = int(round(rd[i], 3) * 1000)
    count = np.bincount(new_rd)
    count_list = np.full(len(count) - 49, 0)
    for i in range(len(count_list)):
        count_list[i] = np.mean(count[i:i + 50])
    mode_min = np.argmax(count_list)
    mode_max = mode_min + 50
    mode = (mode_max + mode_min) / 2
    mode = mode / 1000
    return mode


def gc_correct(rd, gc):
    # correcting gc bias
    bin_count = np.bincount(gc)
    global_rd_ave = np.mean(rd)
    for i in range(len(rd)):
        if bin_count[gc[i]] < 2:
            continue
        mean = np.mean(rd[gc == gc[i]])
        rd[i] = global_rd_ave * rd[i] / mean
    return rd


def distance_matrix(rd_count):
    # calculating euclidean_distances matrix
    rd_count = rd_count.astype(np.float64)
    pos = np.array(range(1, len(rd_count) + 1))
    nr_min = np.min(rd_count)
    nr_max = np.max(rd_count)
    new_pos = (pos - min(pos)) / (max(pos) - min(pos)) * (nr_max - nr_min) + nr_min
    rd_count = rd_count.astype(np.float64)
    new_pos = new_pos.astype(np.float64)
    rd = np.c_[new_pos, rd_count]
    dis = euclidean_distances(rd, rd)
    return dis, new_pos


@njit
def k_matrix(dis, k, bandwidth):
    adap_h = []
    min_matrix = np.zeros((dis.shape[0], k))
    for i in range(dis.shape[0]):
        sort = np.argsort(dis[i])
        k_dist = dis[i][sort[k + 1]] * bandwidth
        adap_h.append(k_dist)
        for j in range(1, k + 1):
            min_matrix[i, j] = sort[j]
    return adap_h, min_matrix


@njit
def calc_kde(dis, min_matrix, adap_h, k):
    # calculating KDE of each bin
    kde = []
    for i in range(min_matrix.shape[0]):
        h = adap_h[i]
        x = np.sum((1 / (2 * math.pi * (h ** 2))) * (np.exp(-((dis[min_matrix[i], i] * dis[min_matrix[i], i]) /
                                                              (2 * (h ** 2))))))
        if x == 0.0:
            cur_kde = 100
        else:
            cur_kde = x / k
        kde.append(cur_kde)
    return kde


def calc_kof_scores(kde, min_matrix, bin_head, k):
    # calculating KOF scores
    scores = np.full(int(len(bin_head)), 0.0)
    for i in range(min_matrix.shape[0]):
        cur_rito = kde[min_matrix[i]] / kde[i]
        cur_sum = np.sum(cur_rito) / k
        scores[i] = cur_sum
    return scores


def scaling_rd(rd, mode):
    posit_rd = rd[rd > mode]
    neg_rd = rd[rd < mode]
    if len(posit_rd) < 50:
        mean_max_rd = np.mean(posit_rd)
    else:
        sort = np.argsort(posit_rd)
        max_rd = posit_rd[sort[-50:]]
        mean_max_rd = np.mean(max_rd)
    if len(neg_rd) < 50:
        mean_min_rd = np.mean(neg_rd)
    else:
        sort = np.argsort(neg_rd)
        min_rd = neg_rd[sort[:50]]
        mean_min_rd = np.mean(min_rd)
    scaling = mean_max_rd / (mode + mode - mean_min_rd)
    for i in range(len(rd)):
        if rd[i] < mode:
            rd[i] /= scaling
    return rd


def seg_rd(rd, bin_head, seg_start, seg_end, seg_count):
    seg_rd = np.full(len(seg_count), 0.0)
    for i in range(len(seg_rd)):
        seg_rd[i] = np.mean(rd[seg_start[i]:seg_end[i]])
        seg_start[i] = bin_head[seg_start[i]] * binSize + 1
        if seg_end[i] == len(bin_head):
            seg_end[i] = len(bin_head) - 1
        seg_end[i] = bin_head[seg_end[i]] * binSize + binSize
    return seg_rd, seg_start, seg_end


def write_data_file(chr, seg_start, seg_end, seg_count, scores):
    """
    write data file
    pos_start, pos_end, lof_score, p_value
    """
    output = open(p_value_file, "w")
    output.write("Chr Num " + '\t' + " Start Position " + '\t' + " End Position " + '\t' + " KOF scores " + '\t\t'
                 + " p value " + '\n')
    for i in range(len(scores)):
        output.write(
            str(chr[i]) + '\t ' + str(seg_start[i]) + ' \t ' + str(seg_end[i]) +
            ' \t ' + str(seg_count[i]) + ' \t ' + str(scores[i]) + '\n')


def write_cnv_file(chr, cnv_start, cnv_end, cnv_type, cn, filename):
    """
    write cnv result file
    pos start, pos end, type, copy number
    """
    output = open(filename, "w")
    for i in range(len(cnv_type)):
        if cnv_type[i] == 2:
            output.write("Chr" + str(chr[i]) + '\t' + str(cnv_start[i]) + '\t' + str(
                cnv_end[i]) + '\t' + str("gain") + '\t' + str(cn[i]) + '\n')
        else:
            output.write("Chr" + str(chr[i]) + '\t' + str(cnv_start[i]) + '\t' + str(
                cnv_end[i]) + '\t' + str("loss") + '\t' + str(cn[i]) + '\n')


def read_seg_file(num_col, num_bin):
    """
    read segment file (Generated by DNAcopy.segment)
    seg file: col, chr, start, end, num_mark, seg_mean
    """
    seg_start = []
    seg_end = []
    seg_count = []
    seg_len = []
    with open("seg", 'r') as f:
        for line in f:
            line_str_list = line.strip().split('\t')
            start = (int(line_str_list[0]) - 1) * num_col + int(line_str_list[2]) - 1
            end = (int(line_str_list[0]) - 1) * num_col + int(line_str_list[3]) - 1
            if start < num_bin:
                if end > num_bin:
                    end = num_bin - 1
                seg_start.append(start)
                seg_end.append(end)
                seg_count.append(float(line_str_list[5]))
                seg_len.append(int(line_str_list[4]))
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    return seg_start, seg_end, seg_count, seg_len


def boxplot(scores):
    four = pd.Series(scores).describe()
    Q1 = four['25%']
    Q3 = four['75%']
    IQR = Q3 - Q1
    upper = Q3 + 0.75 * IQR
    # lower = Q1 - 0.75 * IQR
    return upper


def combining_cnv(seg_chr, seg_start, seg_end, seg_count, scores, upper, mode):
    index = scores > upper
    CNV_chr = seg_chr[index]
    CNV_start = seg_start[index]
    CNV_end = seg_end[index]
    CNV_RD = seg_count[index]
    type = np.full(len(CNV_RD), 1)
    for i in range(len(CNV_RD)):
        if CNV_RD[i] > mode:
            type[i] = 2
    for i in range(len(CNV_RD) - 1):
        if CNV_end[i] + 1 == CNV_start[i + 1] and type[i] == type[i + 1]:
            CNV_start[i + 1] = CNV_start[i]
            type[i] = 0
    index = type != 0
    CNV_RD = CNV_RD[index]
    CNV_chr = CNV_chr[index]
    CNV_start = CNV_start[index]
    CNV_end = CNV_end[index]
    CNV_type = type[index]
    return CNV_chr, CNV_start, CNV_end, CNV_RD, CNV_type


def calculating_copy_number(mode, cnv_rd, cnv_type):
    cn = np.full(len(cnv_type), 0)
    index = cnv_type == 1
    lossRD = cnv_rd[index]
    if len(lossRD) > 2:
        data = np.c_[lossRD, lossRD]
        del_type = KMeans(n_clusters=2, random_state=9).fit_predict(data)
        cnv_type[index] = del_type
        if np.mean(lossRD[del_type == 0]) < np.mean(lossRD[del_type == 1]):
            homo_rd = np.mean(lossRD[del_type == 0])
            hemi_rd = np.mean(lossRD[del_type == 1])
            for i in range(len(cn)):
                if cnv_type[i] == 0:
                    cn[i] = 0
                elif cnv_type[i] == 1:
                    cn[i] = 1
        else:
            hemi_rd = np.mean(lossRD[del_type == 0])
            homo_rd = np.mean(lossRD[del_type == 1])
            for i in range(len(cn)):
                if cnv_type[i] == 1:
                    cn[i] = 0
                elif cnv_type[i] == 0:
                    cn[i] = 1
        purity = 2 * (homo_rd - hemi_rd) / (homo_rd - 2 * hemi_rd)
        for i in range(len(cnv_type)):
            if cnv_type[i] == 2:
                cn[i] = int(2 * cnv_rd[i] / (mode * purity) - 2 * (1 - purity) / purity)
    return cn


# get params
start = datetime.datetime.now()
bam = sys.argv[1]
ref_path = sys.argv[2]
out_path = sys.argv[3]
binSize = int(sys.argv[4])
col = int(sys.argv[5])
k = int(sys.argv[6])
bandwidth = float(sys.argv[7])
path = os.path.abspath('.')
seg_path = path + str("/seg")
p_value_file = out_path + '/' + bam + ".scores.txt"
outfile = out_path + '/' + bam + ".result.txt"

ref = [[] for i in range(23)]
refList = read_bam(bam)
for i in range(len(refList)):
    chr = refList[i]
    chr_num = chr.strip('chr')
    if chr_num.isdigit():
        chr_num = int(chr_num)
        reference = ref_path + '/chr' + str(chr_num) + '.fa'
        ref = read_ref(reference, chr_num, ref)

chrLen = np.full(23, 0)
for i in range(1, 23):
    chrLen[i] = len(ref[i])
RDList, PosList, chrList = bins(ref, binSize, chrLen, bam)
all_chr = []
all_rd = []
all_start = []
all_end = []
modeList = np.full(len(chrList), 0.0)
for i in range(len(chrList)):
    print("analyse " + str(chrList[i]))
    RD = np.array(RDList[i][0])
    pos = np.array(PosList[i][0])
    num_bin = len(RD)
    modeList[i] = mode_rd(RD)
    scale_rd = scaling_rd(RD, modeList[i])
    print("segment count...")
    v = robjects.FloatVector(scale_rd)
    m = robjects.r['matrix'](v, ncol=col)
    robjects.r.source("CBS_data.R")
    robjects.r.CBS_data(m, seg_path)
    num_col = int(num_bin / col) + 1
    seg_start, seg_end, seg_count, seg_len = read_seg_file(num_col, num_bin)
    seg_count = np.array(seg_count)
    seg_count = seg_count[:-1]
    seg_start = seg_start[:-1]
    seg_end = seg_end[:-1]
    seg_count, seg_start, seg_end = seg_rd(RD, pos, seg_start, seg_end, seg_count)
    all_rd.extend(seg_count)
    all_start.extend(seg_start)
    all_end.extend(seg_end)
    all_chr.extend(chrList[i] for j in range(len(seg_count)))

all_chr = np.array(all_chr)
all_start = np.array(all_start)
all_end = np.array(all_end)
all_rd = np.array(all_rd)
for i in range(len(all_rd)):
    if np.isnan(all_rd[i]).any():
        all_rd[i] = (all_rd[i - 1] + all_rd[i + 1]) / 2

# KOF Score
print("Calculating scores...")
dis, new_pos = distance_matrix(all_rd)
dist_k, min_matrix = k_matrix(dis, k, bandwidth)
min_matrix = min_matrix.astype(np.int64)
adap_h = np.reshape(dist_k, (-1, 1))
kde = calc_kde(dis, min_matrix, adap_h, k)
kde = np.array(kde)
kof_scores = calc_kof_scores(kde, min_matrix, all_rd, k)
mode = np.mean(modeList)
write_data_file(all_chr, all_start, all_end, all_rd, kof_scores)
upper = boxplot(kof_scores)
CNV_chr, CNV_start, CNV_end, CNV_rd, CNV_type = combining_cnv(all_chr, all_start, all_end, all_rd, kof_scores, upper,
                                                              mode)
cn = calculating_copy_number(mode, CNV_rd, CNV_type)
write_cnv_file(CNV_chr, CNV_start, CNV_end, CNV_type, cn, outfile)
end = datetime.datetime.now()
print("running time: " + str((end - start).seconds) + " seconds")
