__author__ = "Xavier Hernandez-Alias"
'''
Module for optimizing the codon usage of a sequence based on tissue 
specificities. 
'''
import pkg_resources
import numpy as np
import pandas as pd
import RNA

# LD parameters
linear_design_dir = '/home/suresh/Desktop/bayer/LinearDesign/'
tissue_sp_freq_folder_path = "/home/suresh/Desktop/bayer/CUSTOM/editing_package/data/TS_codon_usage/"
import_from_LD = True
print("This is Suresh's update version...")

# Custom parameters
codon_weights = load_data('data/CUSTOM_codon_weights.csv')
codon_ratios = load_data('data/CUSTOM_tissue_ratios.csv')
codon_freq = load_data('data/CUSTOM_codonfreq_CoCoPuts.csv')
codpair_freq = load_data('data/CUSTOM_codonpairsfreq_CoCoPuts.csv')

# Load data required for optimization
GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}

def load_data(file):
    '''
    Return a dataframe of the required file.

    '''
    stream = pkg_resources.resource_stream(__name__, file)
    return pd.read_csv(stream, index_col=0)


def check_is_optimized(optimizer):
    '''
    Raises an error if the sequence is not optimized.

    Returns
    -------
    None.

    '''
    if not hasattr(optimizer,"pool"):
        raise TypeError("This TissueOptimizer instance is not optimized yet. "
                        "Call 'optimize' with appropriate arguments before "
                        "using this optimizer.")

def relative_codons():
    '''
    Computes normalized codon abundances for each amino-acid family.

    Returns
    -------
    codnorm: DataFrame

    '''
    if "codnorm" not in globals():
        global codnorm
        codnorm = pd.DataFrame()
        AAs = list(set(GENETIC_CODE.values())); AAs.remove("_")
        for species in codon_freq.columns:
            for aa in AAs:
                cod_aa = [c for c in GENETIC_CODE.keys() if GENETIC_CODE[c]==aa]
                max_aa = codon_freq.loc[cod_aa,species].max()
                for cod in cod_aa:
                    codnorm.loc[cod,species] = codon_freq.loc[cod,species]/max_aa

def compute_CPS():
    '''
    Computes the Codon Pair Scores based on the observed vs expected
    frequency. The CPS is + for overrepresented pairs and - for 
    underrepresented pairs. The expected frequency is computed to be 
    independent both of amino acid frequency and of codon bias.

    Returns
    -------
    CPSs: DataFrame

    '''
    if "CPSs" not in globals():
        global CPSs
        CPSs = pd.DataFrame()
        for species in codpair_freq.columns:
            for pair in codpair_freq.index:
                if (pair[0:3] not in ["TAA","TGA","TAG"]) and (pair[3:6] not in ["TAA","TGA","TAG"]):
                    obs_pair = codpair_freq.loc[pair,species]
                    # Compute codon occurrences
                    cod_A = codon_freq.loc[pair[0:3],species]
                    cod_B = codon_freq.loc[pair[3:6],species]
                    # Compute AA occurrences
                    aaA_cods = [c for c in GENETIC_CODE.keys() if GENETIC_CODE[c]==GENETIC_CODE[pair[0:3]]]
                    aaB_cods = [c for c in GENETIC_CODE.keys() if GENETIC_CODE[c]==GENETIC_CODE[pair[3:6]]]
                    aa_A = codon_freq.loc[aaA_cods,species].sum()
                    aa_B = codon_freq.loc[aaB_cods,species].sum()
                    # Compute AA pair occurences
                    aaAB_codpairs = [c1+c2 for c1 in aaA_cods for c2 in aaB_cods]
                    aa_AB = codpair_freq.loc[aaAB_codpairs,species].sum()
                    # Compute expected frequency
                    exp_pair = ((cod_A*cod_B)/(aa_A*aa_B))*aa_AB
                    # Compute CPS
                    CPSs.loc[pair,species] = np.log(obs_pair/exp_pair)

def action(metric,to_do):
    '''
    Takes a list of computed values and converts them to minimize or maximize

    Parameters
    ----------
    metric : ndarray
        Values to optimize
    
    to_do : {"min","max"}
        Min or Max

    Returns
    -------
    norm : ndarray

    '''
    if to_do=="max":
        norm = metric
        return norm
    elif to_do=="min":
        norm = -metric
        return norm
    else:
        raise TypeError("Invalid 'by' argument. Please specify either 'max' "
                        "or 'min' to indicate whether each "
                        "metric should be maximized or minimized.")

        
def filter_sequences(df):
    '''
    Filter sequences: Priority-1, 2 (Description to be updated)
    '''
    def generate_rev_complement(mRNA_sequences):
        rev_comp_list = []
        for mrna_seq in mRNA_sequences:

            # As pair with Us/Ts and Gs pair with Cs. 
            # So rev-compl of ACCTGAG is CTCAGGT 
            # To make it you first write the reverse sequence and then replace each nt with its complement (A <-> U/T and G <-> C)

            # reverse sequence
            rev = mrna_seq[::-1]

            # replace A with X and G with Y
            rev = rev.replace('A','X')
            rev = rev.replace('G','Y')

            # replace T/U with A and C with G
            rev = rev.replace('T','A')
            rev = rev.replace('C','G')

            # replace A with X and G with Y
            rev = rev.replace('X','T')
            rev = rev.replace('Y','C')

            rev_comp_list.append(rev)
        return rev_comp_list   

    def generate_all_mRNA_point_mutants(mRNA_sequences):
        nucleotides = ['A', 'T', 'G', 'C']
        mutants = []

        # Loop through all sequences
        for mrna_seq in mRNA_sequences:

            # Loop through each position in the mRNA sequence
            for position in range(len(mrna_seq)):
                current_nucleotide = mrna_seq[position]

                # Generate a mutant for each substitution at the current position
                for nucleotide in nucleotides:
                    if nucleotide != current_nucleotide:
                        mutant = mrna_seq[:position] + nucleotide + mrna_seq[position + 1:]
                        mutants.append(mutant)

        return mutants
    
    # Priority 1. Internal T7 binding sites, BsaI and SapI RE sites
    # Sequences: TAATACGACTCACTATA, GGTCTC, GCTCTT and rev-compl
    filter_seqs_1 = ['TAATACGACTCACTATA','GGTCTC','GCTCTT']
    rev_compl = generate_rev_complement(filter_seqs_1)
    filter_seqs_1.extend(rev_compl)
    df = df[~df['Sequence'].str.contains('|'.join(filter_seqs_1))]

    # Priority 2. Primer binding sites:
    # Sequences: GATTAAGTTGGGTAACGCCAG (FW), GTAAAACGACGGCCAGT (M13fw), GCATATGACT (polyA-bridge)
    # all 1 point mutant modifications of these, and rev-compl
    filter_seqs_2 = ['GATTAAGTTGGGTAACGCCAG', 'GTAAAACGACGGCCAGT', 'GCATATGACT']
    
    # generate one-point mutants
    one_point_mutants = generate_all_mRNA_point_mutants(filter_seqs_2)
    
    # concat
    filter_seqs_2.extend(one_point_mutants)
    
    # final df
    df = df[~df['Sequence'].str.contains('|'.join(filter_seqs_2))]
    
    return df 

def run_linear_design_tissue_specific(seq, tissue = 'Kidney'):
    import subprocess
    
    # Create the text file and make the input sequence ready
    with open(linear_design_dir+"/testseq", "w") as file:
        file.write('>seq_1\n' + seq + '\n')
        
    # prepare tissue specific codon usage table in the format that linear design takes
    tissue_sp_freq_table = tissue_sp_freq_folder_path + tissue + ".csv"
    
    # run lineardesign tool
    bash_command = "cat testseq | ./lineardesign -l 0.3 --codonusage " + tissue_sp_freq_table
    process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, cwd=linear_design_dir)
    output, error = process.communicate()
    
    # convert results to string format
    ld_results = str(output)
    
    # to parse the output text into a dictionary 
    dict_output = {}

    # extract sequence
    # Find the index of "rmRNA sequence:"
    start_index = ld_results.find("mRNA sequence:  ") + len("mRNA sequence:  ")

    # Find the index of the first space after "rmRNA sequence:"
    end_index = ld_results.find("\\nmRNA", start_index)

    # store
    dict_output['seq'] = ld_results[start_index:end_index]

    # extract mfe
    start_index = ld_results.find("mRNA folding free energy: ")

    # Find the index of the first space after "rmRNA sequence:"
    end_index = ld_results.find("\\n", start_index)

    # Extract the description
    description= ld_results[start_index:end_index]

    # store
    dict_output['description'] = ld_results[start_index:end_index]
    
    return dict_output

class TissueOptimizer:
    '''
    Optimize object, which contains all required methods for tissue-optimizing 
    the codons an amino-acid or nucleotide sequence. Codons are selected based 
    on the Random Forest features that define tissue-specificities, which
    are considered as the probability of optimizing each codon. Directionality
    is based on tissue-specific SDA ratios. Among the pool of generated 
    sequences, the best ones are selected based several commonly used metrics
    (CAI, ENC, CPB, MFE, etc.).
    
    Parameters
    ----------
    tissue: {"Lung","Breast","Skin","Spleen","Heart","Liver","Salivarygland",
             "Muscle...Skeletal","Tonsil","Smallintestine","Placenta",
             "Appendices","Testis","Rectum","Urinarybladder","Prostate",
             "Esophagus","Kidney","Thyroid","Lymphnode","Artery","Brain",
             "Nerve...Tibial","Gallbladder","Uterus","Pituitary","Colon",
             "Vagina","Duodenum","Fat","Stomach","Adrenal","Fallopiantube",
             "Smoothmuscle","Pancreas","Ovary"}
        Tissue to which optimize the sequence
    
    n_pool: int, default=100
        The number of sequences in the generated pool of optimized sequences.
        
    degree: float between 0-1, default=0.5
        Percentage of codons to optimize. Higher values lead to optimizing 
        all codons. Lower values optimize only the most clearly 
        tissue-specific codons.
    
    prob_original: float between 0-1, default=0.0
        Extent to which original codons are conserved. Higher values lead to 
        more conservative optimizations.
    
    Attributes
    ----------
    pool : list
        Minimum Free Energy of optimized sequences. Predicted secondary 
        structures are based on the Vienna RNA Package (Lorenz et al., 2011).
    
    codonprob : dict
        For each amino acid, it contains a nested dictionary matching each
        codon with (1) its tissue-specific weight and (2) its directionality
        (True if its inclusion is favored in that tissue, False otherwise).
    
    sequence : str
        Original sequence to be optimized.
    
    Examples
    --------
    >>> import custom
    >>> opt = TissueOptimizer("kidney", n_pool=50)
    >>> seq = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSG"
    >>> opt.optimize(seq)
    >>> best_seq = opt.select_best(top=10)
    
    '''
    def __init__(self, tissue, n_pool=100, degree=0.5, prob_original=0.0):
        if tissue in codon_weights.index:
            self.tissue = tissue
            self.n_pool = n_pool
            self.degree = degree
            self.prob_original = prob_original
            
            # Get directionality and probabilities for tissue optimization
            # "degree" determines to which extent are unclear codons optimized
            threshold = np.percentile(codon_ratios.loc[self.tissue,:].abs(),(1.0-self.degree)*100)
            codonprob = {}
            for aa in set(GENETIC_CODE.values()):
                aa_codons = [c for c in GENETIC_CODE.keys() if np.logical_and((GENETIC_CODE[c] is aa),(c not in ["ATG","TGG","TAA","TGA","TAG"]))]
                # Avoid deterministic behaviour of codons with weight=0
                if any(codon_weights.loc[self.tissue,aa_codons]):
                    weight_bkg = codon_weights.loc[self.tissue,aa_codons].mean()
                    codweights = {c:codon_weights.loc[self.tissue,c]+weight_bkg for c in aa_codons}
                else:
                    codweights = {c:codon_weights.loc[self.tissue,c] for c in aa_codons}
                sumweights = sum(codweights.values())
                if sumweights==0:
                    sumweights = 1.0
                coddirection = {c:codon_ratios.loc[self.tissue,c]>0 if np.abs(codon_ratios.loc[self.tissue,c])>threshold else np.nan for c in aa_codons}
                codonprob[aa] = {c:[codweights[c]/sumweights,coddirection[c]] for c in aa_codons}
            self.codonprob = codonprob
        else:
            raise TypeError("Invalid 'tissue' argument. Allowed tissues are: "
                            "Lung, Breast, Skin, Spleen, Heart, Liver, Salivarygland, "
                            "Muscle...Skeletal, Tonsil, Smallintestine, Placenta, "
                            "Appendices, Testis, Rectum, Urinarybladder, Prostate, "
                            "Esophagus, Kidney, Thyroid, Lymphnode, Artery, Brain, "
                            "Nerve...Tibial, Gallbladder, Uterus, Pituitary, Colon, "
                            "Vagina, Duodenum, Fat, Stomach, Adrenal, Fallopiantube, "
                            "Smoothmuscle, Pancreas, Ovary.")
    
    def optimize(self, sequence):
        '''
        Create a pool of optimized sequences

        Parameters
        ----------
        sequence : str
            Original nucleotide coding sequence to optimize without stop codon. 
            It also accepts amino acid sequences. In the latter, the 'prob_original' 
            argument is overrun as the optimization runs from scratch with no prior 
            knowledge.

        Returns
        -------
        self : object

        '''
        
        def optaa(self, aa):
            codons = list(self.codonprob[aa].keys())
            newcodon = ""
            while not newcodon:
                if len(codons)==1:
                    newcodon = codons[0]
                else:
                    probs = np.array([self.codonprob[aa][c][0] for c in codons])
                    probs = probs/np.sum(probs)
                    action = np.random.choice(codons,p=probs)
                    if self.codonprob[aa][action][1]==True:
                        newcodon = action
                    elif self.codonprob[aa][action][1]==False:
                        codons.remove(action)
                    else:
                        if np.random.uniform()<=0.5:
                            newcodon = action
                        else:
                            codons.remove(action)
                    
            return newcodon
        
        self.sequence = sequence.upper()
        
        # Identify type of sequence
        if all([s in "ATCG" for s in self.sequence]) and len(self.sequence)%3==0: # DNA sequence
            seqcodons = [self.sequence[n:n+3] for n in range(0,len(self.sequence),3)]
            # Create pool of optimized sequences
            pool = []
            for n in range(self.n_pool):
                newseq = []
                unifrand = np.random.uniform(size=len(seqcodons))
                for n,codon in enumerate(seqcodons):
                    # prob_original allows to take a conservative optimization if wanted
                    if codon in ["ATG","TGG","TAA","TGA","TAG"]: # do not touch stop codons nor ATG or TGG
                        newseq.append(codon)
                    elif unifrand[n]>=self.prob_original:
                        aa = GENETIC_CODE[codon]
                        newseq.append(optaa(self,aa))
                    else:
                        newseq.append(codon)
                pool.append("".join(newseq))
            self.pool = pool
        elif all([s in GENETIC_CODE.values() for s in self.sequence]): # AA sequence
            # Create pool of optimized sequences
            pool = []
            for n in range(self.n_pool):
                newseq = []
                for aa in self.sequence:
                    if aa=="M":
                        newseq.append("ATG")
                    elif aa=="W":
                        newseq.append("TGG")
                    else:
                        newseq.append(optaa(self,aa))
                pool.append("".join(newseq))
            print("Warning: STOP codon not included")
                     
            if import_from_LD:
                # generate new sequence from LinearDesign
                LD_sequence = run_linear_design(self.sequence)['seq']

                # replace U with T to match with other sequences in pool. # Suresh: need to validate
                LD_sequence = LD_sequence.replace('U', 'T')

                # append the sequence generated from LinearDesign
                pool[len(pool)-1] = LD_sequence

                # codon specific linear design
                # generate new sequence from LinearDesign
                LD_sequence = run_linear_design_tissue_specific(self.sequence, tissue = self.tissue)['seq']

                # replace U with T to match with other sequences in pool. # Suresh: need to validate
                LD_sequence = LD_sequence.replace('U', 'T')

                # append the sequence generated from LinearDesign
                pool[len(pool)-2] = LD_sequence
            self.pool = pool
        else:
            raise TypeError("The sequence is not valid.")
    
    def MFE(self):
        '''
        Calculates the Minimum Free Energy of the body (excludes first 40 nt) 
        of optimized sequences.

        Returns
        -------
        MFE: list

        '''
        check_is_optimized(self)
        MFEs = []
        for seq in self.pool:
            # mRNA folding in vivo happens while the CDS is being transcribed.
            # To reproduce it as much as possible, get the average of a 
            # sliding window of 40 nucleotides along the sequence.
            windows = [seq[w:w+40] for w in range(40,(len(seq)-39))]
            mfe_temp = []
            for w in windows:
                mfe_temp.append(RNA.fold(w)[1])
            MFEs.append(np.mean(mfe_temp))
        return MFEs
    
    def MFEini(self):
        '''
        Calculates the Minimum Free Energy of the first 40 nt of optimized 
        sequences.

        Returns
        -------
        MFE: list

        '''
        check_is_optimized(self)
        MFEs = []
        for seq in self.pool:
            MFEs.append(RNA.fold(seq[:40])[1])
        return MFEs
    
    def CAI(self):
        '''
        Calculate the Codon Adaptation Index, based on the similarity to the 
        codon composition of the human genome as a whole (source: CoCoPuts).

        Returns
        -------
        CAI: list

        '''
        check_is_optimized(self)
        relative_codons() # Compute RCU of human genome
        CAIs = []
        for seq in self.pool:
            seqcodons = [seq[n:n+3] for n in range(0,len(seq),3)]
            codus = {c:seqcodons.count(c) for c in set(seqcodons) if c not in ["TAA","TGA","TAG"]}
            cai_codus = [(codnorm.loc[c,"Homo_sapiens"]**(1/sum(codus.values())))**codus[c] for c in codus.keys()]
            CAIs.append(np.prod(cai_codus))
        return CAIs
    
    def CPB(self):
        '''
        Calculate the Codon Pair Bias, based on the similarity to the 
        codon pair composition of the human genome as a whole 
        (source: CoCoPuts).

        Returns
        -------
        CPB: list

        '''
        check_is_optimized(self)
        compute_CPS()
        CPBs = []
        for seq in self.pool:
            seqcodpairs = [seq[n:n+6] for n in range(0,len(seq),3) if (len(seq[n:n+6])==6)and(seq[n+3:n+6] not in ["TAA","TGA","TAG"])]
            codscores = [CPSs.loc[pair,"Homo_sapiens"] for pair in seqcodpairs]
            CPBs.append(np.mean(codscores))
        return CPBs
    
    def ENC(self):
        '''
        Calculate the Effective Number of Codons, which is a measure of codon
        evenness. A value of 20 means that all 100% codons are biased towards 
        the most common codon, while 61 corresponds to no bias at all.

        Returns
        -------
        ENC: list

        '''
        check_is_optimized(self)
        codon_families = {aa:list(GENETIC_CODE.values()).count(aa) for aa in GENETIC_CODE.values() if aa!="_"}
        ENCs = []
        for seq in self.pool:
            seqcodons = [seq[n:n+3] for n in range(0,len(seq),3)]
            codus = {c:seqcodons.count(c) for c in set(seqcodons) if c not in ["TAA","TGA","TAG"]}
            ENC_family = {}
            for f in [2,3,4,6]:
                fam_aa = [aa for aa in codon_families.keys() if codon_families[aa]==f]
                ENCfam = []
                for aa in fam_aa:
                    cod_aa = [c for c in GENETIC_CODE.keys() if GENETIC_CODE[c]==aa]
                    if any([c in codus.keys() for c in cod_aa]):
                        codus_family = np.array([codus[c] if c in codus.keys() else 0.0 for c in cod_aa])
                        codus_sum = codus_family.sum()
                        if codus_sum>1:
                            p_family = codus_family/codus_sum
                            numerator = (codus_sum - 1.0)
                            denominator = (codus_sum * np.sum([p**2 for p in p_family]) - 1.0)
                            if numerator!=0 and denominator!=0:
                                ENCfam.append(numerator/denominator)
                ENC_family[f] = ENCfam
            # Compute contributions of each family
            enc_codus = 2.0 + 9.0*np.mean(ENC_family[2]) + 1.0*np.mean(ENC_family[3]) + 5.0*np.mean(ENC_family[4]) + 3.0*np.mean(ENC_family[6])
            ENCs.append(enc_codus)
        return ENCs
    
    def GC(self):
        '''
        Calculate the GC content of sequences.

        Returns
        -------
        GC: list

        '''
        check_is_optimized(self)
        GCs = []
        for seq in self.pool:
            GCs.append((seq.count("G")+seq.count("C"))/len(seq))
        return GCs
    
    def U_count(self):
        '''
        Calculate the number of Uridine in sequences.
        
        Suresh: Counting T and U for now, Needs revisions

        Returns
        -------
        Us: list

        '''
        check_is_optimized(self)
        Us = []
        for seq in self.pool:
            Us.append((seq.count("T")+seq.count("U"))/len(seq))
        return Us
  
    def UA_count(self):
        '''
        Calculate the ratio of Uracil/Adenine in sequence.
        
        Suresh: Counting T and U both for now, Needs revisions*** 

        Returns
        -------
        UAs: list

        '''
        check_is_optimized(self)
        UAs = []
        for seq in self.pool:
            UAs.append((seq.count("T")+seq.count("U")+seq.count("A"))/len(seq))
        return UAs
    
    def select_best(self, by={"MFE":"min","CAI":"max","CPB":"max","ENC":"min"},
                    homopolymers=0, exclude_motifs = [], top=None):
        '''
        Sort the pool of generated sequences based on different metrics, and
        output the best ones.

        Parameters
        ----------
        by : {"MFE":"min/max","MFEini":"min/max","CAI":"min/max","CPB":"min/max",
              "ENC":"min/max","GC":float}, default={"MFE":"min","CAI":"max","CPB":"max","ENC":"min"}
            Metrics to use in the evaluation of sequences and whether to
            maximize or minimize. For GC, user specified target float
            
        homopolymers : int, default=0
            Removes sequences containing homopolymer sequences of n repeats.
            
        exclude_motifs : list, default=[]
            Removes sequences including certain motifs. Excessively short 
            motifs are recomended against, since the probability of they 
            appearing by chance is high.
            
        top : int or None, default=None
            Output only the top N sequences. If None, all sorted sequences are
            outputted.

        Returns
        -------
        best = DataFrame

        '''
        check_is_optimized(self)
        select_df = pd.DataFrame(index = range(self.n_pool))
        select_df["Sequence"] = self.pool
        norm_df = pd.DataFrame(index = range(self.n_pool))
        for c in by:
            if c=="MFE":
                metric = np.array(self.MFE())
                norm = action(metric,by[c])
            elif c=="MFEini":
                metric = np.array(self.MFEini())
                norm = action(metric,by[c])
            elif c=="CAI":
                metric = np.array(self.CAI())
                norm = action(metric,by[c])
            elif c=="CPB":
                metric = np.array(self.CPB())
                norm = action(metric,by[c])
            elif c=="ENC":
                metric = np.array(self.ENC())
                norm = action(metric,by[c])
            elif c=="GC":
                metric = np.array(self.GC())
                # norm = action(np.abs(metric),"min")  
                norm = action(np.abs(metric-by[c]),"min")
            elif c=="UD":
                metric = np.array(self.U_count())
                # norm = action(np.abs(metric),"min")  
                norm = action(np.abs(metric-by[c]),"min")
            elif c=="UAD":
                metric = np.array(self.UA_count())
                # norm = action(np.abs(metric),"min")  
                norm = action(np.abs(metric-by[c]),"min")
            else:
                raise TypeError("Invalid 'by' argument.")
            select_df[c] = metric
            # Normalize between 0 and 1
            norm_metric = (norm-np.min(norm))/np.ptp(norm)
            norm_df[c] = norm_metric
        # Create a score based on metrics
        select_df["Score"] = norm_df.mean(axis=1)
        # Filter out homopolymers
        if homopolymers>0:
            homo = ["".join([nt]*homopolymers) for nt in ["A","T","C","G"]]
            for h in homo:
                idx = np.array([h in seq for seq in select_df.Sequence])
                select_df = select_df.loc[np.logical_not(idx),:]
        # Filter out motifs
        for m in exclude_motifs:
            idx = np.array([m in seq for seq in select_df.Sequence])
            select_df = select_df.loc[np.logical_not(idx),:]
        
        # filter
        select_df_filtered = filter_sequences(select_df)
        
        # Sort and select best candidates
        best = select_df_filtered.sort_values(by="Score", ascending=False)
        
        if top:
            best = best.head(n=top)
        return best
