import sys
from coffea.nanoevents import NanoEventsFactory
import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import mplhep as hep
import pandas as pd
hep.style.use(hep.style.CMS)
import json

N_exp = {'tttt': 1850, 'ttWl': 33941, 'ttZl': 35563, 'ttWq': 68910, 'ttZq': 82979, 'WZ3l': 275153, 'WZ2l': 855600,
        'ttl': 12199200, 'ttlh': 49128000,'ttH': 29118, 'ZZl':346656, 'ZZq':1213296, 'ttg': 306360, 'tWZ': 415, 'DY1':5244000, 
        'DY2': 173880, 'DY3': 77280}

def read_rootfile(name):
    fname = name
    events = NanoEventsFactory.from_root(fname).events()
    events.genEventSumw = ak.sum(events.genWeight)
    events.LHEScaleSumw = ak.sum(events.genWeight * events.LHEScaleWeight, axis=0) / events.genEventSumw
    genparts = events.GenPart
    return events, genparts

def Get_leadingPart(genpart):
    leading_pts = ak.max(genpart.pt, axis=1)
    cond = (genpart.pt == leading_pts)
    leading_parts = genpart[cond][:, :1]
    return leading_parts 

def Get_ht(events):
    return ak.sum(events.jet.pt, axis=-1) 

def Get_near_t(AK8s):
    parts = events.GenPart
    conds = (np.abs(parts.pdgId) == 6) & (parts.status == 79) 
    ts = parts[conds]
    near_t = AK8s.nearest(ts)
    return near_t

def Get_tDecay(ts):
    children = ak.flatten(ts.children, axis=2)
    condb = np.abs(children.pdgId)==5
    condW = np.abs(children.pdgId)==24
    bs = children[condb]
    Ws = children[condW]
    return bs, Ws

def Get_WDecay(Ws):
    chil = ak.flatten(Ws.children, axis=2)
    children = ak.flatten(chil.children, axis=2)
    cond1 = (children.pdgId < 0 ) & (-10 < children.pdgId)
    cond2 = (children.pdgId > 0) & (10 > children.pdgId)
    q1 = children[cond1]
    q2 = children[cond2]
    mask1 = ak.num(q1)==1
    mask2 = ak.num(q2)==1
    return q1, q2, mask1, mask2

def Jet_tagging(events, Gen_AK8s, r):
    near_ts = Get_near_t(Gen_AK8s)
    b, W = Get_tDecay(near_ts)
    q1, q2, mask1, mask2 = Get_WDecay(W)
    mq1 = ak.mask(q1, mask1)
    mq2 = ak.mask(q2, mask2)
    mAK8 = ak.mask(Gen_AK8s, mask1)

    dR_t = delta_r(near_ts, Gen_AK8s)
    dR_W = delta_r(W, Gen_AK8s)
    dR_b = delta_r(b, Gen_AK8s)
    dR_q1 = delta_r(mq1, mAK8)
    dR_q2 = delta_r(mq2, mAK8)
    
    cond_a = ((dR_t<r) & (dR_W<r) & (dR_b<r) & (dR_q1<r) & (dR_q2<r)) & events.AK8Mask
    cond_b = ((dR_t<r) & (dR_W<r) & (dR_b>r) & (dR_q1<r) & (dR_q2<r)) & events.AK8Mask
    cond_c = (dR_t<r) & (dR_W>r) & (dR_b<r)  & events.AK8Mask
    cond_d = (dR_t>r)  & events.AK8Mask


    # Convert boolean values to lists of length one
    nested_list_a = [[item] if isinstance(item, bool) else item for item in ak.fill_none(cond_a, False, axis=None)]
    nested_list_b = [[item] if isinstance(item, bool) else item for item in ak.fill_none(cond_b, False, axis=None)]
    nested_list_c = [[item] if isinstance(item, bool) else item for item in ak.fill_none(cond_c, False, axis=None)]
    nested_list_d = [[item] if isinstance(item, bool) else item for item in ak.fill_none(cond_d, False, axis=None)]

    # Convert to an awkward array
    nested_array_a = ak.Array(nested_list_a)
    nested_array_b = ak.Array(nested_list_b)
    nested_array_c = ak.Array(nested_list_c)
    nested_array_d = ak.Array(nested_list_d)
    
    events.Flag_a = ak.flatten(nested_array_a)
    events.Flag_b = ak.flatten(nested_array_b)
    events.Flag_c = ak.flatten(nested_array_c)
    events.Flag_d = ak.flatten(nested_array_d)
    
    events.Flag = np.chararray(len(events), unicode=True)
    events.Flag[events.Flag_a] = 'a'
    events.Flag[events.Flag_b] = 'b'
    events.Flag[events.Flag_c] = 'c'
    events.Flag[events.Flag_d] = 'd'

def Get_leading_t(genpart):
    t_cond = (np.abs(genpart.pdgId) == 6) & (genpart.status == 79)
    ts = genpart[t_cond]
    pt_sorted = ak.sort(ts.pt, axis=-1, ascending=False)
    leading_pts = pt_sorted[:, 0]
    cond = (genpart.pt == leading_pts)
    leading_parts = genpart[cond][:, :1]
    return leading_parts    

def Merge_leptons(events):
    merged = ak.concatenate([events.electron, events.muon], axis=1)
    return merged

def Merge_events(mask1, mask2, mask3):
    mask = mask1 | mask2 | mask3
    mask = ak.fill_none(mask, False)
    return mask

def delta_r(a, b):
    def delta_phi(a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi
    def delta_eta(a, b):
        return a-b
    delta_r = np.hypot(delta_eta(a.eta, b.eta), delta_phi(a.phi, b.phi))
    return delta_r

def veto_lepjets(events, jets, dR):
    leps = Merge_leptons(events)
    near_leps = jets.nearest(leps)
    DR = delta_r(jets, near_leps)
    cond = DR > dR
    lepjet_cond = ak.fill_none(cond, True)
    return lepjet_cond

def veto_AK8subjets(events, jets):
    Fatjets = events.FatJet
    near_Fjet = jets.nearest(Fatjets)
    DR = delta_r(jets, near_Fjet)
    cond = DR > 0.4
    subjet_cond = ak.fill_none(cond, True)
    return subjet_cond

def Select_electrons(events):
    el = events.Electron
    pt_cond = el.pt > 10
    n_cond = np.abs(el.eta) < 2.5
    dxy_cond = np.abs(el.dxy) < 0.05
    dz_cond = np.abs(el.dz) < 0.1
    sip3d_cond = el.sip3d < 8
    Irel_cond = el.miniPFRelIso_all < 0.4
    conv_cond = el.convVeto == True
    char_cond = el.tightCharge == 2
    hits_cond = el.lostHits <= 1
    mva_cond = el.mvaTTH > -0.05
    #EGam_cond = el.mvaFall17V2noIso_WPL
    #jets = events.Jet[el.jetIdx]
    #DeepJet_cond = jets.btagDeepFlavB < 0.1
    #PtRatio_cond = 1. / (el.jetRelIso +1) > 0.4
    conds = pt_cond & n_cond & dxy_cond & dz_cond & sip3d_cond & Irel_cond & conv_cond & char_cond & hits_cond & mva_cond
    Obj_sel = el[conds]
    return Obj_sel

def Select_muons(events):
    muon = events.Muon
    pt_cond = muon.pt > 10
    n_cond = np.abs(muon.eta) < 2.4
    dxy_cond = np.abs(muon.dxy) < 0.05
    dz_cond = np.abs(muon.dz) < 0.1
    sip3d_cond = muon.sip3d < 8
    Irel_cond = muon.miniPFRelIso_all < 0.4
    POG_cond = muon.mediumId
    mva_cond = muon.mvaTTH > 0.66
    #jets = events.Jet[muon.jetIdx]
    #DeepJet_cond = jets.btagDeepFlavB < 0.025
    #PtRatio_cond = 1. / (muon.jetRelIso +1) > 0.45
    conds = pt_cond & n_cond & dxy_cond & dz_cond & sip3d_cond & Irel_cond & POG_cond & mva_cond
    Obj_sel = muon[conds]
    return Obj_sel

def Select_jets(events):
    jet = events.Jet
    pt_cond = jet.pt > 25
    n_cond = np.abs(jet.eta) < 2.4
    tight_cond = jet.jetId >= 2
    btag_cond = jet.btagDeepFlavB >= 0.049
    lepjet_cond = veto_lepjets(events, jet, 0.4)
    subjet_cond = veto_AK8subjets(events, jet)
    conds = pt_cond & n_cond & tight_cond & btag_cond & lepjet_cond & subjet_cond
    Obj_sel = jet[conds]
    return Obj_sel

def Select_Fatjets(events):
    Fjet = events.FatJet
    pt_cond = Fjet.pt > 200
    n_cond = np.abs(Fjet.eta) < 2.4
    tight_cond = Fjet.jetId >= 2
    lepjet_cond = veto_lepjets(events, Fjet, 0.8)
    ttag_cond = Fjet.particleNet_TvsQCD > 0.58
    conds = pt_cond & n_cond & tight_cond & lepjet_cond
    tconds = pt_cond & n_cond & tight_cond & lepjet_cond & ttag_cond
    events.tfatjet = Fjet[tconds]
    Obj_sel = Fjet[conds]
    return Obj_sel

def Select_objects(events):
    events.electron = Select_electrons(events)
    events.muon = Select_muons(events)
    events.jet = Select_jets(events)
    events.fatjet = Select_Fatjets(events)
    events.AllMask = ak.num(events.electron) + ak.num(events.muon) >= 1
    return events

def Get_SSleptons(events):
    cond_elel1 = ak.num(events.electron) == 2
    cond_elel2 = (np.sum(events.electron.charge>0, axis=1) == 2) | (np.sum(events.electron.charge<0, axis=1) == 2)
    cond_elel3 = ak.num(events.muon) == 0
    cond_elel = cond_elel1 & cond_elel2 & cond_elel3
    cond_elel = ak.fill_none(cond_elel, False)
    events.elelMask = cond_elel
    
    cond_mumu1 = ak.num(events.muon) == 2
    cond_mumu2 = (np.sum(events.muon.charge>0, axis=1) == 2) | (np.sum(events.muon.charge<0, axis=1) == 2)
    cond_mumu3 = ak.num(events.electron) == 0
    cond_mumu = cond_mumu1 & cond_mumu2 & cond_mumu3
    cond_mumu = ak.fill_none(cond_mumu, False)
    events.mumuMask = cond_mumu

    cond_elmu1 = ak.num(events.electron) == 1
    cond_elmu2 = ak.num(events.muon) == 1
    cond_elmu3 = np.abs(np.sum(events.muon.charge, axis=1) + np.sum(events.electron.charge, axis=1)) == 2
    cond_elmu = cond_elmu1 & cond_elmu2 & cond_elmu3
    cond_elmu = ak.fill_none(cond_elmu, False)
    events.elmuMask = cond_elmu
    return events

def Get_invmass(lep1, lep2):
    vec_one = ak.zip(
        {"pt": lep1.pt, "eta": lep1.eta, "phi": lep1.phi, "mass": 0},with_name="PtEtaPhiMLorentzVector",behavior=vector.behavior)
    vec_two = ak.zip(
        {"pt": lep2.pt, "eta": lep2.eta, "phi": lep2.phi, "mass": 0},with_name="PtEtaPhiMLorentzVector",behavior=vector.behavior)
    return (vec_one+vec_two).mass

def Get_Wleptons(events):
    masked_elel = ak.mask(events.electron, events.elelMask)
    invmass = Get_invmass(masked_elel[:, 0], masked_elel[:, 1])
    cutoff1 = np.abs(invmass - 91) >= 15
    cutoff2 = invmass > 12
    cutoff3 = events.MET.pt > 30
    Welel_cond = cutoff1 & cutoff2 & cutoff3
    Welel_cond = ak.fill_none(Welel_cond, False)
    events.WelelMask = Welel_cond
        
    masked_mumu = ak.mask(events.muon, events.mumuMask)
    invmass = Get_invmass(masked_mumu[:, 0], masked_mumu[:, 1])
    cutoff1 = np.abs(invmass - 91) >= 15
    cutoff2 = invmass > 12
    cutoff3 = events.MET.pt > 30
    Wmumu_cond = cutoff1 & cutoff2 & cutoff3
    Wmumu_cond = ak.fill_none(Wmumu_cond, False)
    events.WmumuMask = Wmumu_cond
    
    masked_events = ak.mask(events, events.elmuMask)
    #invmass = Get_invmass(masked_events.muon[:, 0], masked_events.electron[:, 0])
    #cutoff2 = invmass > 12
    #cutoff3 = masked_events.MET.pt > 30
    #Welmu_cond = cutoff2 & cutoff3
    Welmu_cond = masked_events.MET.pt > 30
    Welmu_cond = ak.fill_none(Welmu_cond, False)
    events.WelmuMask = Welmu_cond
    return events

def Get_AK4Jets(events):
    cutoff = ak.num(events.jet) >= 1
    AK4cond = ak.fill_none(cutoff, False)
    events.elelAK4Mask = AK4cond & events.WelelMask
    events.mumuAK4Mask = AK4cond & events.WmumuMask
    events.elmuAK4Mask = AK4cond & events.WelmuMask
    return events

def Get_AK8Jets(events):
    cutoff = ak.num(events.fatjet) >= 1
    tcutoff = ak.num(events.tfatjet) >= 1
    tJetcond = ak.fill_none(tcutoff, False)
    AK8cond = ak.fill_none(cutoff, False)
    
    events.elelAK8Mask = AK8cond & events.elelAK4Mask
    events.telelAK8Mask = tJetcond & events.elelAK8Mask
    
    events.mumuAK8Mask = AK8cond & events.mumuAK4Mask
    events.tmumuAK8Mask = tJetcond & events.mumuAK8Mask
    
    events.elmuAK8Mask = AK8cond & events.elmuAK4Mask
    events.telmuAK8Mask = tJetcond & events.elmuAK8Mask
    return events

def Get_tJet(events):
    mask = events.FatJet.genJetAK8Idx!=-1
    cond = np.sum(np.abs(events.GenJetAK8.partonFlavour[events.FatJet.genJetAK8Idx[mask]]) == 6, axis=1) == 1 
    cond = ak.fill_none(cond, False)
    TJet_events = ak.mask(events, cond)
    TJet_events.Mask = cond
    return TJet_events

def Select_Jets(events, min_pt, max_pt):
    cond_min = np.sum(events.fatjet.pt > min_pt, axis=1)==1
    cond_max = np.sum(events.fatjet.pt < max_pt, axis=1)==1
    cond = cond_min & cond_max
    cond = ak.fill_none(cond, False)
    return cond

def Select_Events(events):
    SS_events = Get_SSleptons(events)
    W_events = Get_Wleptons(SS_events)
    AK4_events = Get_AK4Jets(W_events)
    AK8_events = Get_AK8Jets(AK4_events)
    events.LepMask = Merge_events(events.elelMask, events.mumuMask, events.elmuMask)
    events.ZMask = Merge_events(events.WelelMask, events.WmumuMask, events.WelmuMask)
    events.AK4Mask = Merge_events(events.elelAK4Mask, events.mumuAK4Mask, events.elmuAK4Mask)
    events.AK8Mask = Merge_events(events.elelAK8Mask, events.mumuAK8Mask, events.elmuAK8Mask)
    events.tAK8Mask = Merge_events(events.telelAK8Mask, events.tmumuAK8Mask, events.telmuAK8Mask)
    events.EvMask = Merge_events(events.elelAK8Mask, events.mumuAK8Mask, events.elmuAK8Mask)
    return events

"""
def get_data(name, events, data, Mask, nbins):
    weights_list = events.genWeight * (N_exp[name] / events.genEventSumw)
    weights_list_duplicated = list(map(lambda x, y: [y] * len(x), data[Mask], weights_list[Mask]))
    hist, bins = np.histogram(ak.flatten(data[Mask]), bins=nbins, weights=ak.flatten(weights_list_duplicated))
    midpoints = (bins[:-1] + bins[1:]) / 2
    return hist, midpoints
"""

inFile = sys.argv[1]
outFile = sys.argv[2]
Process = sys.argv[3]

events, genpart = read_rootfile(inFile)
Obj_selection = Select_objects(events)
lepEv_selection = Select_Events(Obj_selection)

merged_leptons = Merge_leptons(Obj_selection)
leading_parts = Get_leadingPart(merged_leptons)
leading_els = Get_leadingPart(Obj_selection.electron)
leading_mus = Get_leadingPart(Obj_selection.muon)
leading_AK8 = Get_leadingPart(Obj_selection.fatjet)
leading_AK4 = Get_leadingPart(Obj_selection.jet)
near_gent = Get_near_t(leading_AK8)
Ht = Get_ht(events)

#Jet_tagging(events, leading_AK8, 0.8)
#leading_gent = Get_leading_t(genpart)

print(inFile)
print(outFile)
print(len(events))

#data = {'elmu_pt': ak.flatten(leading_parts.pt[events.telmuAK8Mask]), 'genWeights': events.genWeight[events.telmuAK8Mask], 'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}
"""
if len(leading_parts[events.tAK8Mask])>0:
    data = {'lep_pt': ak.flatten(leading_parts.pt[events.tAK8Mask]), 'AK8_pt': ak.flatten(leading_AK8.pt[events.tAK8Mask]), 'H_t': Ht[events.tAK8Mask],
            'AK4_pt': ak.flatten(leading_AK4.pt[events.tAK8Mask]),'t32': ak.flatten(leading_AK8.tau3[events.tAK8Mask]/leading_AK8.tau2[events.tAK8Mask]),
            'Msd': ak.flatten(leading_AK8.msoftdrop[events.tAK8Mask]), 'genWeights': events.genWeight[events.tAK8Mask], 'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}
else:
    data = {'lep_pt': [0], 'AK8_pt': [0], 'H_t': [0], 'AK4_pt': [0], 't32': [0], 'Msd': [0],
            'genWeights': [0], 'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}


if len(leading_parts[events.AK4Mask])>0:
    data = {'lep_pt': ak.flatten(leading_parts.pt[events.AK4Mask]), 'genWeights': events.genWeight[events.AK4Mask],
            'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}
else:
    data = {'lep_pt': [0], 'AK8_pt': [0], 'GenT_pt': [0], 'genWeights': [0], 'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}


iif len(leading_parts[events.tAK8Mask])>0:
    data = {'lep_pt': ak.flatten(leading_parts.pt[events.tAK8Mask]), 'AK8_pt': ak.flatten(leading_AK8.pt[events.tAK8Mask]),
            'LHE_dd': events.LHEScaleWeight[:, 0][events.tAK8Mask], 'LHES_dd': events.LHEScaleSumw[0],
            'LHE_dm':events.LHEScaleWeight[:, 1][events.tAK8Mask], 'LHES_dm': events.LHEScaleSumw[1],
            'LHE_md': events.LHEScaleWeight[:, 3][events.tAK8Mask], 'LHES_md': events.LHEScaleSumw[3],
            'LHE_um': events.LHEScaleWeight[:, 5][events.tAK8Mask], 'LHES_um': events.LHEScaleSumw[5],
            'LHE_mu': events.LHEScaleWeight[:, 7][events.tAK8Mask], 'LHES_mu': events.LHEScaleSumw[7],
            'LHE_uu': events.LHEScaleWeight[:, 8][events.tAK8Mask], 'LHES_uu': events.LHEScaleSumw[8],
            'ISR_u': events.PSWeight[:, 0][events.tAK8Mask], 'ISR_d': events.PSWeight[:, 2][events.tAK8Mask],
            'FSR_u': events.PSWeight[:, 1][events.tAK8Mask], 'FSR_d': events.PSWeight[:, 3][events.tAK8Mask],
            'genWeights': events.genWeight[events.tAK8Mask], 'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}
else:
    data = {'LHE_dd': [0], 'LHE_dm': [0], 'LHE_md': [0], 'LHE_um': [0], 'LHE_mu': [0], 'LHE_uu': [0]}
     
if len(leading_parts[events.AK8Mask])>0:
    data = {'lep_pt': ak.flatten(leading_parts.pt[events.AK8Mask]), 'AK8_pt': ak.flatten(leading_AK8.pt[events.AK8Mask]), 
            't32': ak.flatten(leading_AK8.tau3[events.AK8Mask]/leading_AK8.tau2[events.AK8Mask]), 'Msd': ak.flatten(leading_AK8.msoftdrop[events.AK8Mask]),
            'Flag': events.Flag[events.AK8Mask],
            'genWeights': events.genWeight[events.AK8Mask], 'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}
else:
    data = {'lep_pt': [0], 'AK8_pt': [0], 'GenT_pt': [0], 'genWeights': [0], 'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}

"""

if len(leading_parts[events.ZMask])>0:
    data = {'lep_pt': ak.flatten(leading_parts.pt[events.ZMask]), 'genWeights': events.genWeight[events.ZMask],
            'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}
else:
    data = {'lep_pt': [0],  'genWeights': [0], 'sumWeight': events.genEventSumw, 'N_exp': N_exp[Process]}

print(data)
df = pd.DataFrame(data)

df.to_csv(outFile, sep='\t', index=False)
                                    