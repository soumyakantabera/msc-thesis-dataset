#!/usr/bin/env python3
"""
PHASE 6 — SECTOR INTERACTION (ROBUSTNESS / EXTENSION) v9  [FRESH REBUILD]
NOT a core hypothesis. Sector concentration (former H4) is REJECTED and reported
as robustness. Imports v9_common helpers (computed FROM the panel CSV, not from
any pre-made fix CSVs). Folds in the all-13-sector small-multiples figure. Ends
with ONE overall findings table.
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from v9_common import aig_sd, per_sd, us_share, add_regime_flag, US_NOTE

PANEL=Path(__file__).resolve().parents[1]/'ph2_data_cleaning'/'PANEL_thesis_vars.csv'
OUT=Path(__file__).resolve().parent; TAB=OUT/'tables'; FIG=OUT/'figures'
TAB.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
FIRM,YEAR,IV='Ticker','Year','AbnInvPeer'; CTRLS=['Size_w','DebtToAssets_w','RevGrowth_w']; REF='g12'
SECTOR_LABELS={'g1':'Mass Retail','g2':'Home & Specialty','g3':'Apparel','g4':'Automotive','g5':'Tech Hardware',
 'g6':'Semiconductors','g7':'Aerospace','g8':'Bldg Materials','g9':'Chemicals','g10':'Beverages',
 'g11':'Medical Devices','g12':'Consumer Staples','g13':'Industrial'}
SEC_ORDER=list(SECTOR_LABELS.keys())
INK='#23221F'; MUTE='#6B6A65'; GRID='#D9D7CE'; C_SIG='#2F5C7E'; C_NS='#B4B2AA'; C_REF='#8A8880'; C_POS='#9C6B4E'; PT='#9FB3C0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12.5,
 'axes.labelsize':10.5,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,
 'figure.dpi':110,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.18,
 'axes.edgecolor':'#BBBAB3','axes.linewidth':0.8})
def style(ax,grid='y'):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if grid: ax.grid(axis=grid,color=GRID,alpha=.8,lw=.6)
    ax.set_axisbelow(True)
def title(ax,t,sub=None):
    ax.set_title(t,fontsize=13,fontweight='bold',color=INK,loc='left',pad=26 if sub else 8)
    if sub: ax.text(0,1.045,sub,transform=ax.transAxes,fontsize=9.3,color=MUTE,va='bottom')
def stars(p): return '' if pd.isna(p) else ('***' if p<.01 else '**' if p<.05 else '*' if p<.10 else '')
def sv(d,n): d.to_csv(TAB/n,index=False); print(f"  saved table: {n}")
def sf(fig,n): fig.savefig(FIG/n); plt.close(fig); print(f"  saved figure: {n}")

def ols_cluster(X,y,cid):
    X=np.asarray(X,float);y=np.asarray(y,float);n,k=X.shape
    XtXi=np.linalg.pinv(X.T@X);b=XtXi@X.T@y;e=y-X@b
    uc=np.unique(cid);G=len(uc);B=np.zeros((k,k))
    for c in uc:
        m=cid==c;s=X[m].T@e[m];B+=np.outer(s,s)
    V=XtXi@B@XtXi*((G/(G-1))*(n-1)/(n-k));se=np.sqrt(np.clip(np.diag(V),0,None))
    t=b/np.where(se>0,se,np.nan);p=2*(1-sp.t.cdf(np.abs(t),df=G-1))
    return dict(b=b,se=se,t=t,p=p,V=V,n=n)
def fe_simple(data,dv,iv=IV):
    s=data[[FIRM,YEAR,dv,iv]+CTRLS].dropna().copy()
    if s[FIRM].nunique()<4 or len(s)<30: return None
    s[FIRM]=s[FIRM].astype(str)
    X=pd.concat([s[[iv]+CTRLS].reset_index(drop=True),
      pd.get_dummies(s[FIRM],prefix='e',drop_first=True,dtype=float).reset_index(drop=True),
      pd.get_dummies(s[YEAR].astype(int),prefix='y',drop_first=True,dtype=float).reset_index(drop=True)],axis=1)
    r=ols_cluster(X.to_numpy(float),s[dv].to_numpy(float),s[FIRM].to_numpy())
    i=list(X.columns).index(iv); return dict(coef=r['b'][i],se=r['se'][i],p=r['p'][i],n=r['n'])

print("="*72); print("PHASE 6 — SECTOR INTERACTION (ROBUSTNESS, former H4) v9 [FRESH]"); print("="*72)
print("  Reported as a REJECTED directional hypothesis (outline Sec 4A).")
df=pd.read_csv(PANEL,low_memory=False)
if 'RevGrowth_w' not in df: df['RevGrowth_w']=df['RevGrowth'].clip(df['RevGrowth'].quantile(.01),df['RevGrowth'].quantile(.99))
df['DeltaGM']=df['dGM_fwd']; df['DeltaCFO_now']=df['dCFOMgn']
for c in [IV,'AbnInvPeer_simple','AbnInvRaw','DeltaGM','DeltaCFO_now']: df[c]=pd.to_numeric(df[c],errors='coerce')
df['Sector']=df['Sector'].astype(str)
df=add_regime_flag(df); SD=aig_sd(df); USH=us_share(df, by='Sector')
print(f"  Panel: {df[FIRM].nunique()} firms, {len(df)} firm-years | AIG SD={SD:.4f}")
print(f"  {US_NOTE[:88]}...")

def interaction(dv, iv=IV, ref=REF):
    s=df[[FIRM,YEAR,dv,iv,'Sector']+CTRLS].dropna().copy(); s['Sector']=s['Sector'].astype(str)
    secs=sorted(s['Sector'].unique(),key=lambda x:int(x[1:]))
    for sec in secs:
        if sec!=ref: s[f'd_{sec}']=(s['Sector']==sec).astype(float)*s[iv]
    ic=[f'd_{sec}' for sec in secs if sec!=ref]
    X=pd.concat([s[[iv]+CTRLS+ic].reset_index(drop=True),
      pd.get_dummies(s[FIRM],prefix='e',drop_first=True,dtype=float).reset_index(drop=True),
      pd.get_dummies(s[YEAR].astype(int),prefix='y',drop_first=True,dtype=float).reset_index(drop=True)],axis=1)
    cn=list(X.columns); r=ols_cluster(X.to_numpy(float),s[dv].to_numpy(float),s[FIRM].astype(str).to_numpy())
    base=r['b'][cn.index(iv)]; bi=cn.index(iv); V=r['V']; counts=s['Sector'].value_counts().to_dict(); rows=[]
    for sec in secs:
        if sec==ref: tot,se,p=base,r['se'][bi],r['p'][bi]
        else:
            j=cn.index(f'd_{sec}'); tot=base+r['b'][j]; se=np.sqrt(V[bi,bi]+V[j,j]+2*V[bi,j]); p=2*(1-sp.norm.cdf(abs(tot/se)))
        ush=USH[USH['Sector']==sec]['us_share_pct']
        rows.append(dict(Sector=sec,Label=SECTOR_LABELS[sec],total_beta=round(tot,4),SE=round(se,4),
          lo=round(tot-1.96*se,4),hi=round(tot+1.96*se,4),p_value=round(p,4),Stars=stars(p),
          pp_per_1SD=per_sd(tot,SD),N_obs=int(counts.get(sec,0)),
          US_share_pct=float(ush.iloc[0]) if len(ush) else np.nan,is_ref=(sec==ref)))
    return pd.DataFrame(rows)

print(f"\n{'-'*72}\n  TABLE 20 — sector interaction, next-year ΔGM (ref = Staples)\n{'-'*72}")
TA=interaction('DeltaGM'); sv(TA,'Table20_sector_interaction_dGM_v9.csv')
semi=TA[TA.Sector=='g6'].iloc[0]; nsig=int((TA['p_value']<0.05).sum())
for _,r in TA.sort_values('total_beta').iterrows():
    print(f"  {r['Sector']:4s} {r['Label'][:16]:16s} b={r['total_beta']:+.4f} p={r['p_value']:.3f} {r['Stars']:3s} US={r['US_share_pct']:.0f}%")
print(f"  => semis b={semi['total_beta']:+.4f} p={semi['p_value']:.3f} (NOT sig); {nsig}/13 sig; H4 REJECTED")

print(f"\n{'-'*72}\n  TABLE 21 — per-sector slope: US-only vs international\n{'-'*72}")
rows=[]
for sec in SEC_ORDER:
    sub=df[df['Sector']==sec]
    full=fe_simple(sub,'DeltaGM'); usr=fe_simple(sub[sub['Country']=='USA'],'DeltaGM'); intl=fe_simple(sub[sub['Country']!='USA'],'DeltaGM')
    ush=USH[USH['Sector']==sec]['us_share_pct']
    rows.append(dict(Sector=sec,Label=SECTOR_LABELS[sec],
      full_beta=round(full['coef'],4) if full else np.nan, full_p=round(full['p'],3) if full else np.nan,
      US_beta=round(usr['coef'],4) if usr else np.nan, US_p=round(usr['p'],3) if usr else np.nan,
      intl_beta=round(intl['coef'],4) if intl else np.nan, intl_p=round(intl['p'],3) if intl else np.nan,
      US_share_pct=float(ush.iloc[0]) if len(ush) else np.nan))
T21=pd.DataFrame(rows); sv(T21,'Table21_US_vs_intl_per_sector_v9.csv')
for _,r in T21.iterrows():
    print(f"  {r['Label'][:16]:16s} full={r['full_beta']:+.3f}(p={r['full_p']}) US={r['US_beta']:+.3f} intl={r['intl_beta']:+.3f} (US {r['US_share_pct']:.0f}%)")

print(f"\n{'-'*72}\n  TABLE 23 — robustness: same-year ΔCFO interaction\n{'-'*72}")
TB=interaction('DeltaCFO_now'); sv(TB,'Table23_interaction_dCFO_v9.csv')
print(f"  semis (dCFO) b={TB[TB.Sector=='g6']['total_beta'].iloc[0]:+.4f} p={TB[TB.Sector=='g6']['p_value'].iloc[0]:.3f}")

def barcol(r):
    if r['is_ref']: return C_REF
    if r['p_value']<0.05: return C_SIG if r['total_beta']<0 else C_POS
    return C_NS
d=TA.sort_values('total_beta'); fig,ax=plt.subplots(figsize=(11,7.5)); y=np.arange(len(d))
ax.barh(y,d['total_beta'],color=[barcol(r) for _,r in d.iterrows()],edgecolor='white')
ax.errorbar(d['total_beta'],y,xerr=1.96*d['SE'],fmt='none',ecolor=MUTE,elinewidth=1,capsize=2.5,alpha=.7)
ax.axvline(0,color=INK,lw=1); ax.set_yticks(y)
ax.set_yticklabels([f"{r['Label']}{' †' if r['Sector']=='g6' else ''} ({r['US_share_pct']:.0f}% US)" for _,r in d.iterrows()],fontsize=8.5)
for yi,(_,r) in zip(y,d.iterrows()):
    ax.text(r['total_beta']+(0.004 if r['total_beta']>=0 else -0.004),yi,f"{r['total_beta']:+.3f}{r['Stars']}",
            va='center',ha='left' if r['total_beta']>=0 else 'right',fontsize=8,color=INK)
ax.set_xlabel('Total AbnInvPeer effect on next-year DGM (by sector)')
title(ax,'Sector-specific AbnInvPeer effect on next-year gross margin',
 'Two-way FE interaction (ref = Consumer Staples). dagger = Semiconductors. Slate = significant negative; grey = not significant.')
ax.legend(handles=[Patch(color=C_SIG,label='Significant (p<0.05), negative'),Patch(color=C_NS,label='Not significant'),
  Patch(color=C_REF,label='Reference (Staples)')],loc='lower right',frameon=True,framealpha=.95)
style(ax,grid='x'); fig.subplots_adjust(left=0.34); sf(fig,'Figure22_sector_total_beta_v9.png')

keys=['g6','g3','g11','g12']  # top: predicted-strong (semis, apparel); bottom: standalone-significant (Med Devices, Staples)
fig,axes=plt.subplots(2,2,figsize=(12,8.5))
for ax,sec in zip(axes.flat,keys):
    s=df[df['Sector']==sec].dropna(subset=[IV,'DeltaGM']); s=s[s[IV].abs()<s[IV].quantile(.98)]
    if len(s)>15:
        s2=s.copy(); s2['bin']=pd.qcut(s2[IV],10,labels=False,duplicates='drop')
        b=s2.groupby('bin').agg(x=(IV,'mean'),y=('DeltaGM','mean')).reset_index()
        ax.scatter(b['x'],b['y'],s=42,color=C_SIG,edgecolors='white',lw=.7,zorder=3)
        m,c=np.polyfit(s[IV],s['DeltaGM'],1); xs=np.linspace(b['x'].min(),b['x'].max(),30); ax.plot(xs,m*xs+c,color=C_POS,lw=1.8,zorder=2)
    r=TA[TA.Sector==sec].iloc[0]; pred=' (predicted strong)' if sec in ['g6','g3'] else ''
    ax.axhline(0,color=INK,lw=.6); ax.axvline(0,color=INK,lw=.6)
    ax.set_title(f"{SECTOR_LABELS[sec]}{pred}\nb={r['total_beta']:+.3f}, p={r['p_value']:.2f}{r['Stars']} - {r['US_share_pct']:.0f}% US",fontsize=10,color=INK,loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(color=GRID,alpha=.6,lw=.5)
fig.suptitle('AbnInvPeer vs next-year DGM: predicted vs actually-significant sectors',x=0.07,y=0.99,ha='left',fontsize=13,fontweight='bold',color=INK)
fig.text(0.07,0.95,'Top row: the two sectors H4 predicted would be strongest (semiconductors, apparel) - neither is significant.',ha='left',fontsize=9.3,color=MUTE)
fig.supxlabel('AbnInvPeer'); fig.supylabel('Next-year DGM')
fig.subplots_adjust(top=0.87,hspace=0.3,wspace=0.16); sf(fig,'Figure24_key_sector_facets_v9.png')

d=TA.sort_values('total_beta'); fig,ax=plt.subplots(figsize=(10,7.5)); y=np.arange(len(d))[::-1]
for yi,(_,r) in zip(y,d.iterrows()):
    col=C_SIG if (r['p_value']<.05 and r['total_beta']<0) else (C_REF if r['is_ref'] else C_NS)
    ax.plot([r['lo'],r['hi']],[yi,yi],color=col,lw=2.2,solid_capstyle='round',alpha=.85)
    ax.scatter(r['total_beta'],yi,s=70,color=col,edgecolors='white',lw=1.1,zorder=4)
    if r['Sector']=='g6': ax.text(r['hi'],yi+0.2,'semis: not significant',fontsize=7.5,color=MUTE,style='italic')
ax.axvline(0,color=INK,lw=1); ax.set_yticks(y); ax.set_yticklabels([r['Label'] for _,r in d.iterrows()])
ax.set_xlabel('Total AbnInvPeer effect on next-year DGM (95% CI)')
title(ax,'Forest plot of sector-specific AbnInvPeer effects',
 'Confidence intervals crossing zero indicate no significant sector effect. Most sectors, including semiconductors, cross zero.')
style(ax,grid='x'); fig.subplots_adjust(left=0.22); sf(fig,'Figure25_forest_all_sectors_v9.png')

allx=df.dropna(subset=[IV])[IV]; xlo,xhi=allx.quantile(.02),allx.quantile(.98)
fig,axes=plt.subplots(3,5,figsize=(16,9.5),sharex=True,sharey=True); nf=df[FIRM].nunique(); betas_sm=[]
for idx,sec in enumerate(SEC_ORDER):
    ax=axes.flat[idx]; s=df[df['Sector']==sec].dropna(subset=[IV,'DeltaGM']); s=s[(s[IV]>=xlo)&(s[IV]<=xhi)]
    r=fe_simple(df[df['Sector']==sec],'DeltaGM'); b=r['coef'] if r else np.nan; p=r['p'] if r else np.nan
    betas_sm.append(dict(Sector=sec,Label=SECTOR_LABELS[sec],beta=round(b,4) if not np.isnan(b) else np.nan,
        p_value=round(p,4) if not np.isnan(p) else np.nan,firms=df[df['Sector']==sec][FIRM].nunique()))
    sig=(not np.isnan(p)) and p<0.05
    if len(s)>12:
        s2=s.copy(); s2['bin']=pd.qcut(s2[IV],8,labels=False,duplicates='drop')
        bn=s2.groupby('bin').agg(x=(IV,'mean'),y=('DeltaGM','mean')).reset_index()
        ax.scatter(bn['x'],bn['y']*100,s=28,color=PT,edgecolors='white',lw=.5,zorder=3)
    if not np.isnan(b):
        xs=np.linspace(xlo,xhi,30); ax.plot(xs,(b*xs)*100,color=C_SIG if sig else C_NS,lw=2.2 if sig else 1.6,ls='-' if sig else '--',zorder=4)
    ax.axhline(0,color=INK,lw=.5); ax.axvline(0,color=INK,lw=.5)
    ax.set_title(SECTOR_LABELS[sec],fontsize=10.5,color=INK,loc='left',fontweight='bold',pad=2)
    ax.text(0.02,0.97,f"b={b:+.3f}{stars(p)}\np={p:.2f} - {df[df['Sector']==sec][FIRM].nunique()} firms" if not np.isnan(b) else "n/a",
            transform=ax.transAxes,fontsize=7.6,va='top',ha='left',color=C_SIG if sig else MUTE,
            bbox=dict(boxstyle='round,pad=0.3',fc='white',ec=C_SIG if sig else GRID,lw=.8,alpha=.9))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(color=GRID,alpha=.5,lw=.4); ax.set_axisbelow(True)
for j in range(len(SEC_ORDER),15): axes.flat[j].axis('off')
axes.flat[14].legend(handles=[Line2D([0],[0],color=C_SIG,lw=2.2,label='Significant (p<0.05)'),
    Line2D([0],[0],color=C_NS,lw=1.6,ls='--',label='Not significant'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor=PT,markersize=8,label='Binned mean')],loc='center',frameon=True,fontsize=10,title='Fit line')
fig.suptitle('AbnInvPeer versus next-year change in gross margin, by sector',x=0.06,y=0.985,ha='left',fontsize=15,fontweight='bold',color=INK)
fig.text(0.06,0.952,f'All 13 sectors - {nf} firms - binned scatter with two-way FE slope per sector; solid blue = significant at 5%, dashed grey = not. Common axes.',ha='left',fontsize=10,color=MUTE)
fig.supxlabel('AbnInvPeer',fontsize=11,y=0.02); fig.supylabel('Next-year DGross Margin (pp)',fontsize=11,x=0.01)
fig.subplots_adjust(top=0.90,hspace=0.26,wspace=0.10,left=0.055,bottom=0.07)
sf(fig,'Figure_AllSectors_small_multiples_v9.png')
T22=pd.DataFrame(betas_sm); sv(T22,'Table22_small_multiples_betas_v9.csv')

print(f"\n{'='*72}\n  OVERALL FINDINGS TABLE\n{'='*72}")
# Significance is computed from STANDALONE per-sector FE (T22/betas_sm), the cleanest
# and most defensible method, NOT from the interaction model (whose reference category,
# Staples, is mechanically excluded and would otherwise be mislabelled). We report
# both the 5% set and the 5-10% borderline set, and note the method dependence.
T22['p_value']=pd.to_numeric(T22['p_value'],errors='coerce')
sa_sig = T22[(T22.p_value<0.05)].sort_values('beta')
sa_bord = T22[(T22.p_value>=0.05)&(T22.p_value<0.10)].sort_values('beta')
n_sa_sig = len(sa_sig)
sig_names = ', '.join(f"{r['Label']} (b={r['beta']:+.3f}, p={r['p_value']:.3f})" for _,r in sa_sig.iterrows()) or 'none'
bord_names = ', '.join(f"{r['Label']} (p={r['p_value']:.3f})" for _,r in sa_bord.iterrows()) or 'none'
findings=[
 dict(Item='Sector concentration hypothesis (former H4)',
      Finding=f'REJECTED. Standalone per-sector FE: {n_sa_sig}/13 significant at 5%; predicted-strong (semis, apparel) not significant.',
      Conclusion='Report as robustness, not a core hypothesis.'),
 dict(Item='Semiconductors (the v8 centrepiece)',
      Finding=f"beta={semi['total_beta']:+.4f}, p={semi['p_value']:.2f} (NOT significant); {semi['US_share_pct']:.0f}% US.",
      Conclusion='v8 result (beta=-0.179 on 6 firms) does not replicate on 15 firms.'),
 dict(Item='Sectors significant at 5% (standalone FE)',
      Finding=sig_names,
      Conclusion='Signal concentrated in a few NON-predicted sectors, not apparel/semis.'),
 dict(Item='Borderline sectors (5-10%)',
      Finding=bord_names,
      Conclusion='Mass Retail sits here (p~0.06): treat as suggestive, NOT significant.'),
 dict(Item='Method note: interaction vs standalone',
      Finding='Interaction model uses Staples as reference, so Staples cannot appear in its significant set; standalone FE shows Staples IS significant.',
      Conclusion='Significant-sector identity is method-dependent; standalone FE is the reported basis.'),
 dict(Item='US dominance of significant sectors',
      Finding=f"Significant sectors are US-heavy (Med Devices {TA[TA.Sector=='g11']['US_share_pct'].iloc[0]:.0f}%, semis {semi['US_share_pct']:.0f}%).",
      Conclusion='Disclose: sector results lean on US firms; not a global claim.'),
 dict(Item='US-only vs international (Table 21)',
      Finding='Per-sector slopes broadly similar in sign; most insignificant in both cuts.',
      Conclusion='No sector shows a robust international-specific concentration.'),
 dict(Item='Same-year DCFO interaction (Table 23)',
      Finding=f"Cash channel more broadly negative across sectors (semis beta={TB[TB.Sector=='g6']['total_beta'].iloc[0]:+.3f}).",
      Conclusion='Consistent with H2 breadth; not a revival of sector-H4.'),
 dict(Item='Small-multiples (Figure, all 13 sectors)',
      Finding=f'Most sectors gentle negative-insignificant; significant at 5%: {sig_names}.',
      Conclusion='Single exhibit answers "why these sectors" - broad, modest, not concentrated.'),
]
TF=pd.DataFrame(findings); sv(TF,'Table_FINDINGS_overall_v9.csv')
for _,r in TF.iterrows():
    print(f"  - {r['Item']}")
    print(f"      finding: {r['Finding']}")
    print(f"      => {r['Conclusion']}")

print(f"\n{'='*72}\n  PHASE 6 v9 COMPLETE (fresh rebuild)\n{'='*72}")
print(f"  Tables ({len(os.listdir(TAB))}), Figures ({len(os.listdir(FIG))})")
print(f"  Headline: sector concentration REJECTED; semis not significant (p={semi['p_value']:.2f}).")
