import sys
from coffea.nanoevents import NanoEventsFactory
import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import mplhep as hep
hep.style.use(hep.style.CMS)
import json

N_exp = {'tttt': 1850, 'ttWl': 33941, 'ttZl': 35563, 'ttWq': 68910, 'ttZq': 82979, 'WZ3l': 275153, 'WZ2l': 855600,
        'ttl': 12199200, 'ttH': 29118, 'ZZl':346656, 'ZZq':1213296, 'ttg': 306360, 'tWZ': 415, 'DY1':5244000, 'DY2': 173880, 'DY3': 77280}

def read_rootfile(name):
    fname = name
    events = NanoEventsFactory.from_root(fname).events()
    events.genEventSumw = ak.sum(events.genWeight)
    genparts = events.GenPart
    return events, genparts

def Get_leadingPart(genpart):
    leading_pts = ak.max(genpart.pt, axis=1)
    cond = (genpart.pt == leading_pts)
    leading_parts = genpart[cond][:, :1]
    return leading_parts  

def Merge_leptons(events):
    merged = ak.concatenate([events.electron, events.muon], axis=1)
    return merged

def Merge_events(mask1, mask2):
    mask = mask1 | mask2
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
    lepjet_cond = ak.fill_none(cond, False)
    return lepjet_cond

def veto_AK8subjets(events, jets):
    Fatjets = events.FatJet
    near_Fjet = jets.nearest(Fatjets)
    DR = delta_r(jets, near_Fjet)
    cond = DR > 0.4
    subjet_cond = ak.fill_none(cond, False)
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
    ttag_cond = Fjet.particleNet_TvsQCD > 0.5
    conds = pt_cond & n_cond & tight_cond & lepjet_cond
    tconds = pt_cond & n_cond & tight_cond & lepjet_cond & ttag_cond
    Obj_sel = Fjet[tconds]
    return Obj_sel

def Select_objects(events):
    events.electron = Select_electrons(events)
    events.muon = Select_muons(events)
    events.jet = Select_jets(events)
    events.fatjet = Select_Fatjets(events)
    events.AllMask = ak.num(events.electron) + ak.num(events.muon) >= 1
    return events

def Get_OSElectrons(events):
    cond_el1 = ak.num(events.electron) == 2 #number of electrons is 2
    cond_el2 = np.sum(events.electron.charge>0, axis=1) == 1 #The electrons have an opposite sign
    cond_muon = ak.num(events.muon) == 0 #number of leptons should be exactly 2
    OSSFel_conds = cond_el1 & cond_el2 & cond_muon #conditions in order to have opposite sign, same flavour lepton events
    events.ElMask = OSSFel_conds
    return events

def Get_OSMuons(events):
    cond_muon1 = ak.num(events.muon) == 2 #number of muons is 2
    cond_muon2 = np.sum(events.muon.charge>0, axis=1) == 1 #muons should have opposite sign
    cond_el = ak.num(events.electron) == 0 #number of leptons should be exactly 2
    OSSFmu_conds = cond_el & cond_muon1 & cond_muon2 #conditions in order to have opposite sign, same flavour lepton events
    events.MuMask = OSSFmu_conds
    return events

def Get_invmass(parts):
    lep1 = parts[:, 0]
    lep2 = parts[:, 1]
    vec_one = ak.zip(
        {"pt": lep1.pt, "eta": lep1.eta, "phi": lep1.phi, "mass": 0},with_name="PtEtaPhiMLorentzVector",behavior=vector.behavior)
    vec_two = ak.zip(
        {"pt": lep2.pt, "eta": lep2.eta, "phi": lep2.phi, "mass": 0},with_name="PtEtaPhiMLorentzVector",behavior=vector.behavior)
    return (vec_one+vec_two).mass

def Get_Zelectrons(events):
    masked_electrons = ak.mask(events.electron, events.ElMask)
    invmass = Get_invmass(masked_electrons)
    cutoff = np.abs(invmass - 91) <= 10
    Zcond = ak.fill_none(cutoff, False)
    events.ElZMask = Zcond & events.ElMask
    return events

def Get_Zmuons(events):
    masked_muons = ak.mask(events.muon, events.MuMask)
    invmass = Get_invmass(masked_muons)
    cutoff = np.abs(invmass - 91) <= 10
    Zcond = ak.fill_none(cutoff, False)
    events.MuZMask = Zcond & events.MuMask 
    return events

def Get_AK4Jets(events, lep):
    cutoff = ak.num(events.jet) >= 1
    AK4cond = ak.fill_none(cutoff, False)
    if lep == 'el':
        events.ElAK4Mask = AK4cond & events.ElZMask
    if lep == 'mu':
        events.MuAK4Mask = AK4cond & events.MuZMask
    return events

def Get_AK8Jets(events, lep):
    cutoff = ak.num(events.fatjet) >= 1
    AK8cond = ak.fill_none(cutoff, False)
    if lep == 'el':
        events.ElAK8Mask = AK8cond & events.ElAK4Mask
    if lep == 'mu':
        events.MuAK8Mask = AK8cond & events.MuAK4Mask
    return events

def Get_tJet(events):
    mask = events.FatJet.genJetAK8Idx != -1
    cond = np.sum(np.abs(events.GenJetAK8.partonFlavour[events.FatJet.genJetAK8Idx[mask]]) == 6, axis=1) == 1 
    cond = ak.fill_none(cond, False)
    return cond

def Select_Jets(events, min_pt, max_pt):
    cond_min = np.sum(events.fatjet.pt > min_pt, axis=1)==1
    cond_max = np.sum(events.fatjet.pt < max_pt, axis=1)==1
    cond = cond_min & cond_max & events.AK8Mask
    cond = ak.fill_none(cond, False)
    return cond

def Select_ElectronEvents(events):
    OSElectrons = Get_OSElectrons(events)
    Zlepton_events = Get_Zelectrons(OSElectrons)
    AK4_events = Get_AK4Jets(Zlepton_events, 'el')
    AK8_events = Get_AK8Jets(AK4_events, 'el')
    return AK8_events

def Select_MuonEvents(events):
    OSMuons = Get_OSMuons(events)
    Zlepton_events = Get_Zmuons(OSMuons)
    AK4_events = Get_AK4Jets(Zlepton_events, 'mu')
    AK8_events = Get_AK8Jets(AK4_events, 'mu')
    return AK8_events

def Select_Events(events):
    elel_events = Select_ElectronEvents(events)
    mumu_events = Select_MuonEvents(events)
    events.LepMask = Merge_events(events.ElMask, events.MuMask)
    events.ZMask = Merge_events(events.ElZMask, events.MuZMask)
    events.AK4Mask = Merge_events(events.ElAK4Mask, events.MuAK4Mask)
    events.AK8Mask = Merge_events(events.ElAK8Mask, events.MuAK8Mask)
    events.EvMask = Merge_events(events.ElAK8Mask, events.MuAK8Mask)
    return events

def get_data(name, events, data, Mask, nbins):
    weights_list = events.genWeight * (N_exp[name] / events.genEventSumw)
    weights_list_duplicated = list(map(lambda x, y: [y] * len(x), data[Mask], weights_list[Mask]))
    hist, bins = np.histogram(ak.flatten(data[Mask]), bins=nbins, weights=ak.flatten(weights_list_duplicated))
    midpoints = (bins[:-1] + bins[1:]) / 2
    return hist, midpoints


inFile = sys.argv[1]
outFile = sys.argv[2]
Process = sys.argv[3]

events, genpart = read_rootfile(inFile)
Obj_selection = Select_objects(events)
lepEv_selection = Select_Events(Obj_selection)

merged_leptons = Merge_leptons(Obj_selection)
leading_parts = Get_leadingPart(merged_leptons)
leading_els = Get_leadingPart(Obj_selection.electron)
leading_Els = Get_leadingPart(events.Electron)
leading_mus = Get_leadingPart(Obj_selection.muon)
leading_AK8 = Get_leadingPart(Obj_selection.fatjet)

hist, midpoints = get_data(Process, events, leading_parts.pt, events.EvMask, np.linspace(0, 400, 8))


print(outFile)
print(hist)
print(len(events))

with open(outFile,'w') as o:
    hist_list = hist.tolist()
    o.write(json.dumps(hist_list))
    o.write('\n')
    o.write(str(len(events)))
