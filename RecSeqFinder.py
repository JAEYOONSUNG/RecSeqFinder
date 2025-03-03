import os
import pandas as pd
import functools
from Bio.Seq import Seq
from snapgene_reader import snapgene_file_to_dict
import traceback

# Define IUPAC ambiguous DNA values
ambiguous_dna_values = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "M": "AC", "R": "AG", "W": "AT", "S": "CG", "Y": "CT", "K": "GT",
    "V": "ACG", "H": "ACT", "D": "AGT", "B": "CGT", "N": "GATC"
}

# Default restriction enzyme dictionary (case-sensitive keys)
default_restriction_enzymes = {
"AatII": "GACGTC",
"AbaDI": "C",
"AbaHI": "C",
"AbaSI": "C",
"AbrI": "CTCGAG",
"AcaPI": "C",
"AccI": "GTMKAC",
"I-AchMI": "TGGGGAGGTTTTTCAGTATC",
"AciI": "CCGC",
"AclI": "AACGTT",
"Aco12261II (RM.Aco12261II)": "CCRGAG",
"Afa22MI": "CGATCG",
"AflII": "CTTAAG",
"AflIII": "ACRYGT",
"AgeI": "ACCGGT",
"AhdI": "GACNNNNNGTC",
"C.AhdI": "ACTCATAGTCCGTGGACTTATCGA",
"AloI (RM.AloI)": "GAACNNNNNNTCC",
"AluI": "AGCT",
"Alw26I": "GTCTC",
"Aod1I (RM.Aod1I)": "GATCNAC",
"ApaI": "GGGCCC",
"ApeKI": "GCWGC",
"AplI": "CTGCAG",
"ApyPI (RM.ApyPI)": "ATCGAC",
"AquI": "CYCGRG",
"AquII (RM.AquII)": "GCCGNAC",
"AquIII (RM.AquIII)": "GAGGAG",
"AquIV (RM.AquIV)": "GRGGAAG",
"AscI": "GGCGCGCC",
"AseI": "ATTAAT",
"AspBHI": "YSCNS",
"AteTI (RM.AteTI)": "GGGRAG",
"AvaI": "CYCGRG",
"AvaII": "GGWCC",
"AvaIII": "ATGCAT",
"BalI": "TGGCCA",
"BamFI": "GGATCC",
"BamHI": "GGATCC",
"C.BamHI": "ACTTATAGTCTGTAGCCTATAGTC",
"Ban4602I (RM.Ban4602I)": "GACGAG",
"BanLI": "RTCAGG",
"I-BasI": "AACGCTCAGCAATTCCCACGT",
"Bbr215I (RM.Bbr215I)": "GGCGAG",
"Bbr7017II (RM.Bbr7017II)": "CGGGAG",
"R1.BbrUI": "GGCGCC",
"BbrUII": "GTCGAC",
"BbrUIII": "CTGCAG",
"BbvI": "GCAGC",
"BbvCI (BbvCIA)": "CCTCAGC",
"BbvCI (BbvCIB)": "CCTCAGC",
"Bce95I": "GCNGC",
"Bce1273I": "GCNGC",
"Bce3081I (RM.Bce3081I)": "TAGGAG",
"Bce14579I (Bce14579IB)": "GCWGC",
"Bce14579I (Bce14579IA)": "GCWGC",
"BceJI": "CACAG",
"BceSI": "CGAAG",
"BceSII": "GGWCC",
"BceSIII": "ACGGC",
"BceSIV": "GCAGC",
"BceYI": "GCNGC",
"BcgI (RM.BcgI)": "CGANNNNNNTGC",
"BclI": "TGATCA",
"BcnI": "CCSGG",
"BfaI (BfaIA)": "CTAG",
"BfaI (BfaIB)": "CTAG",
"BfaSII (RM.BfaSII)": "GANGGAG",
"BfiI": "ACTGGG",
"BfuAI": "ACCTGC",
"BglI": "GCCNNNNNGGC",
"BglII": "AGATCT",
"C.BglII": "ACTTATAGTCCGTGGACACATAGT",
"BhaI": "GCATC",
"BhaII": "GGCC",
"Nt.BhaIII": "GAGTC",
"BisI": "GCNGC",
"BloAII (RM.BloAII)": "GAGGAC",
"BmeDI": "C",
"I-BmoI": "GAGTAAGAGCCCGTAGTAATGACATGGC",
"BmtI": "GCTAGC",
"BpmI (RM.BpmI)": "CTGGAG",
"Bpu10I (Bpu10IA)": "CCTNAGC",
"Bpu10I (Bpu10IB)": "CCTNAGC",
"BpuJI": "CCCGT",
"BsaAI": "YACGTR",
"BsaHI": "GRCGYC",
"BsaJI": "CCNNGG",
"Bse634I": "RCCGGY",
"BseMII (RM.BseMII)": "CTCAG",
"BseYI (BseYIA)": "CCCAGC",
"BseYI (BseYIB)": "CCCAGC",
"BsgI (RM.BsgI)": "GTGCAG",
"BslI (BslIA)": "CCNNNNNNNGG",
"BslI (BslIB)": "CCNNNNNNNGG",
"BsmI": "GAATGC",
"BsmAI": "GTCTC",
"BsmBI": "CGTCTC",
"BsmFI (RM.BsmFI)": "GGGAC",
"Bsp6I": "GCNGC",
"Bsp98I": "GGATCC",
"Bsp460III (RM.Bsp460III)": "CGCGCAG",
"BspCNI (RM.BspCNI)": "CTCAG",
"Nt.BspD6I": "GAGTC",
"BspEI": "TCCGGA",
"BspHI": "TCATGA",
"BspLU11III (RM.BspLU11III)": "GGGAC",
"BspQI": "GCTCTTC",
"BspRI": "GGCC",
"BsrI": "ACTGG",
"BsrDI (BsrDIA)": "GCAATG",
"BsrDI (BsrDIB)": "GCAATG",
"Nb.BsrDI": "GCAATG",
"BsrFI": "RCCGGY",
"BssHII": "GCGCGC",
"Nt.BstNBI": "GAGTC",
"Nt.BstSEI": "GAGTC",
"BstVI": "CTCGAG",
"BstXI": "CCANNNNNNTGG",
"BstYI": "RGATCY",
"BstZ1II": "AAGCTT",
"Bsu3610I (RM.Bsu3610I)": "GACGAG",
"Bsu7003I (RM.Bsu7003I)": "GACGAG",
"BsuBI": "CTGCAG",
"BsuMI (BsuMIA)": "CTCGAG",
"BsuMI (BsuMIB)": "CTCGAG",
"BsuMI (BsuMIC)": "CTCGAG",
"BsuRI": "GGCC",
"I-BthII": "ATTATCCGTGATGAGTCAATTCA",
"Bth171I": "GCNGC",
"I-Bth0305I": "ATGTAAACTCACGCTTCGGTGATCCAAACGTGACAACTG",
"Btr192II": "ACATC",
"BtsI (BtsIA)": "GCAGTG",
"BtsI (BtsIB)": "GCAGTG",
"BtsCI": "GGATG",
"Bve1B23I (RM.Bve1B23I)": "GACNNNNNTGG",
"Cac824I": "GCNGC",
"Cal14237I (RM.Cal14237I)": "GGTTAG",
"CbeI": "GGCC",
"CceI": "CCGG",
"Cce743I": "GACGC",
"Cce743II": "CCWGG",
"CchI": "CTAG",
"CchII (RM.CchII)": "GGARGA",
"CchIII (RM.CchIII)": "CCCAAG",
"CcoMI (RM.CcoMI)": "CAGCAG",
"CcrNAIII (RM.CcrNAIII)": "CGACCAG",
"CdpI (RM.CdpI)": "GCGGAG",
"Cdu23823II (RM.Cdu23823II)": "GTGAAG",
"I-CeuI": "TAACTATAACGGTCCTAAGGTAGCGAA",
"Cfr9I": "CCCGGG",
"Cfr10I": "RCCGGY",
"Cfr42I": "CCGCGG",
"CfrBI": "CCWWGG",
"CfrCI": "C",
"CglI": "GCSGC",
"Cgl13032II (RM.Cgl13032II)": "ACGABGG",
"I-ChuI": "GAAGGTTTGGCACCTCGATGTCGGCTCATC",
"CjeI (RM.CjeI)": "CCANNNNNNGT",
"CjeFIII (RM.CjeFIII)": "GCAAGG",
"CjeFIV": "GCANNNNNRTTA",
"CjeFV (RM.CjeFV)": "GGRCA",
"CjeNII (RM.CjeNII)": "GAGNNNNNGT",
"CjeNIII (RM.CjeNIII)": "GKAAYG",
"CjeNIV": "GCANNNNNRTTA",
"ClaI": "ATCGAT",
"Cla11845III (RM.Cla11845III)": "GCGAA",
"Cly7489II (RM.Cly7489II)": "AAAAGRG",
"I-CpaI": "CGATCCTAAGGTAGCGAAATTCA",
"I-CpaII": "CCCGGCTAACTCTGTGCCAG",
"CphBI": "CCCGGG",
"I-CreI": "CTGGGTTCAAAACGTCGTGAGACAGTTTGG",
"Csp231I": "AAGCTT",
"Csp68KI": "GGWCC",
"Csp68KII": "TTCGAA",
"Csp68KIII": "ATGCAT",
"Csp68KVI": "CGCG",
"CstMI (RM.CstMI)": "AAGGAG",
"CviAI": "GATC",
"CviAII": "CATG",
"CviJI": "RGCY",
"Nt.CviPII": "CCD",
"CviQI": "GTAC",
"Nt.CviQII": "RAG",
"Nt.CviQXI": "RAG",
"I-CvuI": "CTGGGTTCAAAACGTCGTGAGACAGTTTGG",
"DdeI": "CTNAG",
"Dde51507I": "CCWGG",
"I-DdiI": "TTTTTTGGTCATCCAGAAGTATAT",
"DdsI": "GGATCC",
"I-DmoI": "ATGCCTTGCCGGGTAAGTTCCGGCGCGCAT",
"DpnI": "GATC",
"DpnII": "GATC",
"DraI": "TTTAAA",
"DraRI (RM.DraRI)": "CAAGNAC",
"DrdI": "GACNNNNNNGTC",
"DrdIV (RM.DrdIV)": "TACGAC",
"Dsp20I": "GCWGC",
"DvuI": "TGACNNNNNNTTC",
"EacI": "GGATC",
"EaeI": "YGGCCR",
"EagI": "CGGCCG",
"Ecl18kI": "CCNGG",
"Eco31I": "GGTCTC",
"Eco47II": "GGNCC",
"Eco57I (RM.Eco57I)": "CTGAAG",
"Eco128I": "CCWGG",
"EcoAI": "GAGNNNNNNNGTCA",
"EcoAO83I": "GGANNNNNNNNATGC",
"EcoEI": "GAGNNNNNNNATGC",
"EcoGIII": "CTGCAG",
"EcoGVIII": "ACCACC",
"EcoHK31I": "YGGCCR",
"EcoHSI (RM.EcoHSI)": "GGTAAG",
"EcoKI": "AACNNNNNNGTGC",
"EcoKMcrA": "YCGR",
"EcoMVII (RM.EcoMVII)": "CANCATC",
"EcoO109I": "RGGNCCY",
"C.EcoO109I": "CTAAGNNNNNCTTAG",
"EcoPI": "AGACC",
"EcoP15I": "CAGCAG",
"ECORI": "GAATTC",
"EcoRV": "GATATC",
"EcoR124I": "GAANNNNNNRTCG",
"EcoR124II": "GAANNNNNNNRTCG",
"F-EcoT5I": "TGGCGACGAAAACCGCTTGGAAAGTGGCTG",
"F-EcoT5II": "ACCTACCATTAACGGAGTCAAAGGCCATTG",
"F-EcoT5IV": "TAGGTACTGGACTTAAAATTCAGGTTTTGT",
"EcoT38I": "GRGCYC",
"EcoVIII": "AAGCTT",
"Eco29kI": "CCGCGG",
"EcoprrI": "CCANNNNNNNRTGC",
"ElmI": "GCNGC",
"EsaWC1I": "GGCC",
"Esp3I": "CGTCTC",
"Esp638I": "GCNGC",
"Esp1396I": "CCANNNNNTGG",
"C.Esp1396I": "ATGTGACTTATAGTCCGTGTGATTATAGTCAACAT",
"FnuDI": "GGCC",
"Fnu4HI": "GCNGC",
"FokI": "GGATG",
"FseI": "GGCCGGCC",
"FspI": "TGCGCA",
"FspEI": "CC",
"Fsp4HI": "GCNGC",
"FssI": "GGWCC",
"FtnUIV": "GATC",
"FtnUV (RM.FtnUV)": "GAAACA",
"GauT27I (RM.GauT27I)": "CGCGCAGG",
"GmeII": "TCCAGG",
"I-GzeI": "GCCCCTCATAACCCGTATCAAG",
"HaeII": "RGCGCY",
"HaeIII": "GGCC",
"HaeIV (RM.HaeIV)": "GAYNNNNNRTC",
"HauII (RM.HauII)": "TGGCCA",
"HgiBI": "GGWCC",
"HgiCI": "GGYRCC",
"HgiCII": "GGWCC",
"HgiDI": "GRCGYC",
"HgiDII": "GTCGAC",
"HgiEI": "GGWCC",
"HhaI": "GCGC",
"HhaII": "GANTC",
"Hin4II": "CCTTC",
"HinP1I": "GCGC",
"HincII": "GTYRAC",
"HindII": "GTYRAC",
"HindIII": "AAGCTT",
"HinfI": "GANTC",
"I-HmuI": "AACGCTCAGCAATTCCCACGT",
"I-HmuII": "ATTATTCCAGTATAACTTTGAGATTAAG",
"HphI": "GGTGA",
"Hpy51I": "GTSAC",
"Hpy99I": "CGWCG",
"Hpy99II": "GTSAC",
"Hpy99III": "GCGC",
"Hpy99IV": "CCNNGG",
"Hpy99XIII (RM.Hpy99XIII)": "GCCTA",
"Hpy99XIV (RM.Hpy99XIV)": "GGWTAA",
"Hpy99XIV-mut1 (RM.Hpy99XIV-mut1)": "GGWCNA",
"Hpy99XVI": "RTAYNNNNNRTAY",
"Hpy99XXII (RM.Hpy99XXII)": "TCANNNNNNTRG",
"Hpy188I": "TCNGA",
"Hpy300VIII (RM.Hpy300VIII)": "AGGAG",
"Hpy300XI (RM.Hpy300XI)": "CCTYNA",
"HpyAII": "GAAGA",
"HpyAIII": "GATC",
"HpyAIV": "GANTC",
"HpyAV": "CCTTC",
"HpyAXIV (RM.HpyAXIV)": "GCGTA",
"HpyAXVI-mut1 (RM.HpyAXVI-mut1)": "CRTTAA",
"HpyAXVI-mut2 (RM.HpyAXVI-mut2)": "CRTCNA",
"HpyAXVIII (HpyAXVIIIA) (RM.HpyAXVIII)": "GGANNAG",
"HpyAXVIII (HpyAXVIIIB) (RM.HpyAXVIII)": "GGANNAG",
"HpyAXVIII (HpyAXVIIIC) (RM.HpyAXVIII)": "GGANNAG",
"HpyC1I": "CCATC",
"HpyGI": "TCNNGA",
"HpyHI": "CTNAG",
"HpyUM032XIII (RM.HpyUM032XIII)": "CYANNNNNNNTRG",
"HpyUM032XIV (RM.HpyUM032XIV)": "GAAAG",
"HsoI": "GCGC",
"KasI": "GGCGCC",
"KpnI": "GGTACC",
"Kpn2I": "TCCGGA",
"C.Kpn2I": "TTTTGATACAAAATCATATTAAAAATATGACTCCT",
"Kpn156V (RM.Kpn156V)": "CRTGATT",
"KpnAI": "GGCANNNNNNTTC",
"KpnBI": "CAAANNNNNNRTCA",
"Kpn2kI": "CCNGG",
"I-LlaI": "CACATCCATAACCATATCATTTTT",
"LlaBI": "CTRYAG",
"LlaBIII (RM.LlaBIII)": "TNAGCC",
"LlaCI": "AAGCTT",
"LlaDII": "GCNGC",
"LlaDCHI": "GATC",
"LlaGI (RM.LlaGI)": "CTNGAYG",
"LlaKR2I": "GATC",
"LlaMI": "CCNGG",
"Lmo370I (RM.Lmo370I)": "AGCGCCG",
"Lmo540I (RM.Lmo540I)": "TAGRAG",
"Lmo6907II (RM.Lmo6907II)": "TAGRAG",
"Lpl116III (RM.Lpl116III)": "CAGRAG",
"LpnPI": "CCDG",
"Lsp1109I": "GCAGC",
"I-LtrI": "TATCTAAACGTCGTATAGGAGC",
"I-LtrWI": "AGTAGTGAAGTATGTTATTTAATTCG",
"MamI": "GATNNNNATC",
"MaqI (RM.MaqI)": "CRTTGAC",
"Mba11I (RM.Mba11I)": "AGGCGA",
"MboII": "GAAGA",
"McaCI": "CCATC",
"McaTI": "GCGCGC",
"MchCM4I (RM.MchCM4I)": "GAGGAG",
"Mha185III (RM.Mha185III)": "CTGAAG",
"MjaI": "CTAG",
"MjaII": "GGNCC",
"MjaIII": "GATC",
"MjaIV": "GTNNAC",
"MjaV": "GTAC",
"MluI": "ACGCGT",
"MmeI (RM.MmeI)": "TCCRAC",
"MmeII": "GATC",
"MmyCI": "TGAG",
"MnlI": "CCTC",
"I-MpeMI": "TAGATAACCATAAGTGGCTAAT",
"MscI": "TGGCCA",
"MseI": "TTAA",
"I-MsoI": "CTGGGTTCAAAACGTCGTGAGACAGTTTGG",
"MspI": "CCGG",
"MspA1I": "CMGCKG",
"MspJI": "CNNR",
"Mte37I": "C",
"MthTI": "GGCC",
"MthZI": "CTAG",
"PI-MtuI": "AACGCGGTCGGCAACCGCACCCGGGTCAC",
"MunI": "CAATTG",
"C.MunI": "GCTTATAGTCCGTTGTTTTTAAAC",
"MvaI": "CCWGG",
"Mva1261III": "CTANNNNNNRTTC",
"Mva1269I": "GAATGC",
"MwoI": "GCNNNNNNNGC",
"NaeI": "GCCGGC",
"I-NanI": "AAGTCTGGTGCCAGCACCCGC",
"Nar7I (RM.Nar7I)": "CTGRAG",
"NciI": "CCSGG",
"NcoI": "CCATGG",
"NcuI": "GAAGA",
"NflHI (RM.NflHI)": "GCGGAG",
"NgoAI": "RGCGCY",
"NgoAIII": "CCGCGG",
"NgoAIV": "GCCGGC",
"NgoAV": "GCANNNNNNNNTGC",
"NgoAVI": "GATC",
"NgoAVII": "GCCGC",
"NgoAVIII (RM.NgoAVIII)": "GACNNNNNTGA",
"NgoAX": "CCACC",
"NgoBI": "RGCGCY",
"NgoBV": "GGNNCC",
"NgoBVIII": "GGTGA",
"NgoFVII": "GCSGC",
"NgoMIII": "CCGCGG",
"NgoMIV": "GCCGGC",
"NgoMVIII": "GGTGA",
"NgoPII": "GGCC",
"NhaXI (RM.NhaXI)": "CAAGRAG",
"NheI": "GCTAGC",
"I-NitI": "AAGTCTGGTGCCAGCACCCGC",
"I-NjaI": "AAGTCTGGTGCCAGCACCCGC",
"NlaIV": "GGNNCC",
"NmeAII": "GATC",
"NmeAIII (RM.NmeAIII)": "GCCGAG",
"NmeBI": "GACGC",
"NmeBL859I": "GATC",
"NmeDI": "RCCGGY",
"NmeSI": "AGTACT",
"NpeUS61II (RM.NpeUS61II)": "GATCGAC",
"NsoJS138I": "CAGCTG",
"NspI": "RCATGY",
"NspV": "TTCGAA",
"NspHI": "RCATGY",
"ObaBS10I (RM.ObaBS10I)": "ACGAG",
"OgrI (RM.OgrI)": "CAACNAC",
"I-OnuI": "GGTTGAATAAGTGG",
"PabI": "GTAC",
"PacIII (RM.PacIII)": "GTAATC",
"Pac25I": "CCCGGG",
"PaeR7I": "CTCGAG",
"Pam7686I": "CCATGG",
"Pan13I": "GCWGC",
"I-PanMI": "GCTCCTCATAATCCTTATCAAG",
"PatTI": "C",
"PcaII (RM.PcaII)": "GACGAG",
"PcoI (RM.PcoI)": "GAACNNNNNNTCC",
"PflMI": "CCANNNNNTGG",
"PfrCI": "C",
"PfrJS12IV (RM.PfrJS12IV)": "TANAAG",
"PhoI": "GGCC",
"PI-PkoI": "GATTTTAGATCCCTGTACC",
"PI-PkoII": "CAGTACTACGGTTAC",
"PlaDI (RM.PlaDI)": "CATCAG",
"PluTI": "GGCGCC",
"PmeI": "GTTTAAAC",
"I-PnoMI": "TGAGGTGGTTTCTCTGTAAATTAA",
"I-PogTE7I": "CTTCAGTATGCCCCGAAAC",
"PpeHI": "C",
"I-PpoI": "TAACTATGACTCTCTTAAGGTAGCCAAAT",
"Pps170I": "GCWGC",
"Pru4541I": "GCWGC",
"PI-PspI": "TGGCAAACAGCTATTATGGGTATTATGGGT",
"PspGI": "CCWGG",
"PspOMII (RM.PspOMII)": "CGCCCAR",
"Pst14472I (RM.Pst14472I)": "CNYACAC",
"PvuI": "CGATCG",
"PxyI": "C",
"RceI (RM.RceI)": "CATCGAC",
"RdeGBI (RM.RdeGBI)": "CCGCAG",
"RdeGBII (RM.RdeGBII)": "ACCCAG",
"RdeGBIII (RM.RdeGBIII)": "TGRYCA",
"RdeR2I": "GCNGC",
"RlaI": "VCW",
"RlaII (RM.RlaII)": "ACACAG",
"RpaI (RM.RpaI)": "GTYGGAG",
"RpaBI (RM.RpaBI)": "CCCGCAG",
"RpaB5I (RM.RpaB5I)": "CGRGGAC",
"RpaTI (RM.RpaTI)": "GRTGGAG",
"RsaI": "GTAC",
"RshI": "CGATCG",
"RsrI": "GAATTC",
"Saf8902III (RM.Saf8902III)": "CAATNAG",
"SalI": "GTCGAC",
"Sau96I": "GGNCC",
"Sau3AI": "GATC",
"SauN315I": "AGGNNNNNGAT",
"SauNewI": "SCNGS",
"SauUSI": "SCNGS",
"SbaI": "CAGCTG",
"SbfI": "CCTGCAGG",
"ScaI": "AGTACT",
"I-ScaI": "TGTCACATTGAGGTGCACTAGTTATTAC",
"F-SceI": "GATGCTGTAGGCATAGGCTTGGTT",
"I-SceI": "TAGGGATAACAGGGTAAT",
"PI-SceI": "ATCTATGTCGGGTGCGGAGAAAGAGGTAATGAAATGG",
"I-SceII": "TTTTGATTCTTTGGTCACCCTGAAGTATA",
"I-SceIII": "ATTGGAGGTTTTGGTAACTATTTATTACC",
"I-SceVI": "GTTATTTAATGTTTTAGTAGTTGG",
"ScrFI": "CCNGG",
"SdaI": "CCTGCAGG",
"Sde240I": "GCNGC",
"Sen1736II (RM.Sen1736II)": "GATCAG",
"SenAZII": "CAGAG",
"SenTFIV (RM.SenTFIV)": "GATCAG",
"SenpCI": "CCGCGG",
"SfaNI": "GCATC",
"SfcI": "CTRYAG",
"SfeI": "CTRYAG",
"SfiI": "GGCCNNNNNGGCC",
"SgeI": "CNNG",
"Sgr13350I": "GAGCTC",
"SgrAI": "CRCCGGYG",
"SgrTI": "CCDS",
"SinI": "GGWCC",
"SmaI": "CCCGGG",
"C.SmaI": "ACTCATCGTCTGTCGACTTATAGT",
"I-SmaMI": "GGTATCCTCCATTATCAGGTGTACG",
"SmoLI": "CCGG",
"SmoLII (RM.SmoLII)": "GCAGT",
"SnaBI": "TACGTA",
"Sno506I (RM.Sno506I)": "GGCCGAG",
"SonI": "ATCGAT",
"SphI": "GCATGC",
"SpnD39IIIA": "CRAANNNNNNNNCTG",
"SpnD39IIIB": "CRAANNNNNNNNNTTC",
"SpnD39IIIC": "CACNNNNNNNCTT",
"SpnD39IIID": "CACNNNNNNNCTG",
"SpoDI (RM.SpoDI)": "GCGGRAG",
"I-SscMI": "AGGTACCCTTTAAACCTATTAA",
"Sse9I": "AATT",
"SsoI": "GAATTC",
"SsoII": "CCNGG",
"SspI": "AATATT",
"I-Ssp6803I": "GTCGGGCTCATAACCCGAA",
"Ssp6803IV (RM.Ssp6803IV)": "GAAGGC",
"SstE37I (RM.SstE37I)": "CGAAGAC",
"SstE37III (RM.SstE37III)": "CTGAAG",
"Ssu211I": "GATC",
"Ssu212I": "GATC",
"Ssu220I": "GATC",
"R1.Ssu2479I": "GATC",
"R2.Ssu2479I": "GATC",
"R1.Ssu4109I": "GATC",
"R2.Ssu4109I": "GATC",
"R1.Ssu4961I": "GATC",
"R2.Ssu4961I": "GATC",
"R1.Ssu8074I": "GATC",
"R2.Ssu8074I": "GATC",
"R1.Ssu11318I": "GATC",
"R2.Ssu11318I": "GATC",
"R1.SsuDAT1I": "GATC",
"R2.SsuDAT1I": "GATC",
"Sth368I": "GATC",
"StsI": "GGATG",
"StyI": "CCWWGG",
"StyD4I": "CCNGG",
"StyLTI": "CAGAG",
"StySBLI": "GGTANNNNNNTCG",
"StySKI": "CGATNNNNNNNGTTA",
"SuaI": "GGCC",
"Sve396I": "GCWGC",
"TaqII (RM.TaqII)": "GACCGA",
"F-TevI": "GAAACACAAGAAATGTTTAGTAAA",
"I-TevI": "AGTGGTATCAACGCTCAGTAGATG",
"F-TevII": "TTTAATCCTCGCTTCAGATATGGCAACTG",
"I-TevII": "GCTTATGAGTATGAAGTGAACACGTTATTC",
"I-TevIII": "TATGTATCTTTTGCGTGTACCTTTAACTTC",
"TfiI": "GAWTC",
"TfiTok6A1I": "TCGA",
"TflI": "TCGA",
"PI-TfuI": "TAGATTTTAGGTCGCTATATCCTTCC",
"PI-TfuII": "TAYGCNGAYACNGACGGYTTYT",
"ThaI": "CGCG",
"TliI": "CTCGAG",
"TmaI": "CGCG",
"TneDI": "CGCG",
"I-TslI": "GTACATGGCGTGGGTGCATGATGAAATCCAAGTAGGTTGC",
"I-TslWI (I-TslWI.AY769990)": "GTACATGGCGTGGGTGCATGATGAGATTCAAGTAGGTTGC",
"TsoI (RM.TsoI)": "TARCCA",
"Tsp32I": "TCGA",
"Tsp45I": "GTSAC",
"Tsp509I": "AATT",
"TspDTI": "ATGAA",
"TspGWI (RM.TspGWI)": "ACGGA",
"TspMI": "CCCGGG",
"TspRI": "CASTG",
"TstI (RM.TstI)": "CACNNNNNNTCC",
"Tth111I": "GACNNNGTC",
"Tth111II (RM.Tth111II)": "CAARCA",
"TthHB8I": "TCGA",
"TthHB27I (RM.TthHB27I)": "CAARCA",
"UbaLAI": "CCWGG",
"I-Vdi141I": "CCTGACTCTCTTAAGGTAGCCAAA",
"Vtu19109I (RM.Vtu19109I)": "CACRAYC",
"XamI": "GTCGAC",
"XbaI": "TCTAGA",
"XcmI": "CCANNNNNNNNNTGG",
"XcyI": "CCCGGG",
"XhoI": "CTCGAG",
"XmnI": "GAANNNNTTC",
"XorKI": "CGATCG",
"XorKII": "CTGCAG",
"XphI": "CTGCAG",
"YkrI": "C"
}

# Create case-insensitive restriction enzyme dictionary
restriction_enzymes = {}
for key, value in default_restriction_enzymes.items():
    restriction_enzymes[key] = value
    restriction_enzymes[key.upper()] = value

def sequence_comparison(dna_seq, RM_seq, column_name, reverse_match_count, is_circular=False):
    """Count restriction enzyme recognition sequences in DNA
    - dna_seq: DNA sequence (Seq object)
    - RM_seq: Recognition sequence
    - column_name: Output column name
    - reverse_match_count: Count of simultaneous matches in forward and reverse strands
    - is_circular: Whether the DNA is circular
    - Returns: forward count, reverse count, simultaneous match count, unique site count
    """
    dna_seq_RC = dna_seq.reverse_complement()
    identity_score = 0  # Forward strand matches
    identity_score_RC = 0  # Reverse strand matches
    seq_len = len(dna_seq)
    rm_len = len(RM_seq)
    unique_positions = set()  # Track unique positions

    # Basic linear counting
    for i in range(seq_len):
        subject_frag = dna_seq[i:i+rm_len]
        subject_frag_RC = dna_seq_RC[i:i+rm_len]
        if len(subject_frag) < rm_len:
            break
        
        match_forward = functools.reduce(lambda x, y: x and y,
                                         map(lambda p, q: True if (p == q) else p in ambiguous_dna_values[q],
                                             list(subject_frag), list(RM_seq)), True)
        if match_forward:
            identity_score += 1
            unique_positions.add(i)
            print(f"Forward match at {i}: {subject_frag}")

        match_reverse = functools.reduce(lambda x, y: x and y,
                                         map(lambda p, q: True if (p == q) else p in ambiguous_dna_values[q],
                                             list(subject_frag_RC), list(RM_seq)), True)
        if match_reverse:
            identity_score_RC += 1
            unique_positions.add(i)
            print(f"Reverse match at {i}: {subject_frag_RC}")

        if match_forward and match_reverse:
            reverse_match_count[column_name] += 1

    # Circular DNA boundary correction
    if is_circular:
        for i in range(seq_len - rm_len + 1, seq_len):
            overlap = seq_len - i
            subject_frag = dna_seq[i:] + dna_seq[:rm_len - overlap]
            subject_frag_RC = dna_seq_RC[i:] + dna_seq_RC[:rm_len - overlap]

            match_forward = functools.reduce(lambda x, y: x and y,
                                             map(lambda p, q: True if (p == q) else p in ambiguous_dna_values[q],
                                                 list(subject_frag), list(RM_seq)), True)
            if match_forward and i not in unique_positions:
                identity_score += 1
                unique_positions.add(i)
                print(f"Circular boundary match at {i}: {subject_frag}")

            match_reverse = functools.reduce(lambda x, y: x and y,
                                             map(lambda p, q: True if (p == q) else p in ambiguous_dna_values[q],
                                                 list(subject_frag_RC), list(RM_seq)), True)
            if match_reverse and i not in unique_positions:
                identity_score_RC += 1
                unique_positions.add(i)
                print(f"Circular boundary RC match at {i}: {subject_frag_RC}")

            if match_forward and match_reverse:
                reverse_match_count[column_name] += 1

    total_unique = len(unique_positions)
    print(f"Total unique positions for {column_name}: {total_unique}")
    return identity_score, identity_score_RC, reverse_match_count, total_unique

# Get restriction sequence input from user
RM_seq_input = input("Enter RM recognition sequences (comma-separated, e.g., GATC, EcoRI): ")
RM_seq_list_input = [seq.strip() for seq in RM_seq_input.split(",")]

# Convert enzyme names to recognition sequences
RM_seq_list = []
column_names = []
for seq in RM_seq_list_input:
    print(f"Processing input: {seq}")
    if seq in restriction_enzymes:
        rec_seq = restriction_enzymes[seq]
        RM_seq_list.append(rec_seq)
        column_names.append(f"{rec_seq} ({seq})")
        print(f"Recognized restriction enzyme {seq}: {rec_seq}")
    elif seq.upper() in restriction_enzymes:
        rec_seq = restriction_enzymes[seq.upper()]
        RM_seq_list.append(rec_seq)
        column_names.append(f"{rec_seq} ({seq})")
        print(f"Recognized restriction enzyme {seq.upper()}: {rec_seq}")
    else:
        RM_seq_list.append(seq)
        column_names.append(seq)
        print(f"Treating {seq} as a raw sequence")

# Get working directory from user
working_directory = input("Enter the working directory path: ").strip()
working_directory = working_directory.replace("'", "").replace('"', "")
working_directory = os.path.abspath(os.path.expanduser(working_directory))

if not os.path.isdir(working_directory):
    raise FileNotFoundError(f"Directory not found: {working_directory}")

# Recursively find all .dna files
dna_file_paths = []
for root, _, files in os.walk(working_directory):
    for file in files:
        if file.endswith(".dna"):
            dna_file_paths.append(os.path.join(root, file))

print(f"Total DNA files found: {len(dna_file_paths)}")

# Initialize data lists
dna_file_paths_processed = []
dna_names = []
dna_seq_length = []
data_F = []
data_R = []
data_M = []
reverse_match_count_dna = {}
skip_list = []

# Process DNA files
for file_path in dna_file_paths:
    try:
        print(f"Attempting to parse: {file_path}")
        dictionary = snapgene_file_to_dict(file_path)
        if 'seq' not in dictionary:
            raise ValueError("No 'seq' key in SnapGene dictionary")
        
        forward_seq = Seq(dictionary['seq']).upper()
        current_length = len(forward_seq)
        is_circular = dictionary.get('is_circular', False)
        print(f"{file_path} is {'circular' if is_circular else 'linear'}, Length: {current_length}")

        if current_length > 100000:
            print(f"WARNING: Skipping {file_path}: DNA sequence too long (>100,000 bp)")
            skip_list.append((file_path, "Too long"))
            continue

        reverse_match_count = {col: 0 for col in column_names}
        data_sub_F = []
        data_sub_R = []
        data_sub_M = []
        for rm_seq, col_name in zip(RM_seq_list, column_names):
            print(f"Comparing sequence with {rm_seq} ({col_name})")
            result_F, result_R, reverse_match_count, total_unique = sequence_comparison(
                forward_seq, rm_seq, col_name, reverse_match_count, is_circular
            )
            data_sub_F.append(result_F)
            data_sub_R.append(result_R)
            data_sub_M.append(total_unique)

        current_name = os.path.basename(file_path)[:-4]
        dna_file_paths_processed.append(file_path)
        dna_names.append(current_name)
        dna_seq_length.append(current_length)
        data_F.append(data_sub_F)
        data_R.append(data_sub_R)
        data_M.append(data_sub_M)
        reverse_match_count_dna[file_path] = reverse_match_count.copy()

        print(f"Processed {current_name}: F={data_sub_F}, R={data_sub_R}, M={data_sub_M}, ReverseMatch={reverse_match_count}")

    except Exception as e:
        print(f"WARNING: Error processing {file_path}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        skip_list.append((file_path, str(e)))
        continue

# Create dataframes for processed files
if dna_file_paths_processed:
    print(f"Creating dataframes with {len(dna_file_paths_processed)} files")
    
    df_F = pd.DataFrame(data_F, index=dna_file_paths_processed, columns=column_names)
    df_R = pd.DataFrame(data_R, index=dna_file_paths_processed, columns=column_names)
    df_M = pd.DataFrame(data_M, index=dna_file_paths_processed, columns=column_names)
    df_reverse_match = pd.DataFrame.from_dict(reverse_match_count_dna, orient='index', columns=column_names)

    for df in [df_F, df_R, df_M, df_reverse_match]:
        if 'Length (bp)' not in df.columns:
            df.insert(0, 'Length (bp)', dna_seq_length)

    df_F_out = df_F.set_index(pd.Index(dna_names))
    df_R_out = df_R.set_index(pd.Index(dna_names))
    df_M_out = df_M.set_index(pd.Index(dna_names))
    df_reverse_match_out = df_reverse_match.set_index(pd.Index(dna_names))

    df_M_out.index.name = 'File Name'
    df_F_out.index.name = 'File Name'
    df_R_out.index.name = 'File Name'
    df_reverse_match_out.index.name = 'File Name'

    sort_columns_with_index = column_names + ['File Name']
    df_M_out = df_M_out.sort_values(by=sort_columns_with_index, ascending=[False] * len(RM_seq_list) + [True])

    df_F_out = df_F_out.loc[df_M_out.index]
    df_R_out = df_R_out.loc[df_M_out.index]
    df_reverse_match_out = df_reverse_match_out.loc[df_M_out.index]

    output_file = os.path.join(working_directory, "result.xlsx")
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    # Changed sheet name from 'All count' to 'Total count'
    df_M_out.to_excel(writer, sheet_name='Total count')
    df_F_out.to_excel(writer, sheet_name='Forward Read')
    df_R_out.to_excel(writer, sheet_name='Reverse Read')
    df_reverse_match_out.to_excel(writer, sheet_name='Reverse Match Count')
    
    writer.close()
    print(f"Results saved to: {output_file}")
else:
    print("No valid DNA files were processed successfully.")

if skip_list:
    print("WARNING: The following files were skipped during processing:")
    for path, error in skip_list:
        print(f"  - File: {path}, Error: {error}")
