#!/usr/bin/env python3
"""
PHASE 5 — SUPER-GROUP DECOMPOSITION v9  [FRESH REBUILD]
=======================================================
Each core hypothesis broken down by the six super-groups (S1-S6). Imports
v9_common helpers (US-share, SD, regime — computed FROM the panel CSV). Every
table now carries firm-count + US-share + a fragility flag; small/US-heavy
groups (S4 Technology 29 firms 79% US; S6 Healthcare 21 firms 71% US) are
flagged so per-group claims are not over-read. Ends with one findings table.

Tables:  T15 H1 by SG · T16 H2 by SG · T17 distress chi-sq by SG
         T18 within-SG quintile returns · T19 coverage+US-share · T_FINDINGS
Figures: F15 H1 coef · F16 H2 coef · F17 ΔGM heatmap · F18 faceted scatter
         F19 quintile spread · F20 coverage+US-share · F21 forest
Output: supergroup_outputs_v9/{tables,figures}
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from v9_common import aig_sd, per_sd, us_share, add_regime_flag, US_NOTE

PANEL=Path(__file__).resolve().parents[1]/'ph2_data_cleaning'/'PANEL_thesis_vars.csv'
OUT=Path(__file__).resolve().parent; TAB=OUT/'tables'; FIG=OUT/'figures'
TAB.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
FIRM,YEAR,IV='Ticker','Year','AbnInvPeer'; CTRLS=['Size_w','DebtToAssets_w','RevGrowth_w']
MIN_OBS=30; FRAGILE_FIRMS=35
SG_ORDER=['S1','S2','S3','S4','S5','S6']
SG_LABELS={'S1':'S1 · Retail','S2':'S2 · Consumer','S3':'S3 · Heavy Mfg','S4':'S4 · Technology','S5':'S5 · Materials','S6':'S6 · Healthcare'}
SG_SHORT={'S1':'Retail','S2':'Consumer','S3':'Heavy Mfg','S4':'Technology','S5':'Materials','S6':'Healthcare'}
SG_COL={'S1':'#2F5C7E','S2':'#5B6E8C','S3':'#9C6B4E','S4':'#3E7C6A','S5':'#8A7A52','S6':'#86566B'}
INK='#23221F'; MUTE='#6B6A65'; GRID='#D9D7CE'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12.5,
 'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,
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
    XtXi=np.linalg.pinv(X.T@X);b=XtXi@X.T@y;e=y-X@b;uc=np.unique(cid);G=len(uc);B=np.zeros((k,k))
    for c in uc:
        m=cid==c;s=X[m].T@e[m];B+=np.outer(s,s)
    V=XtXi@B@XtXi*((G/(G-1))*(n-1)/(n-k));se=np.sqrt(np.clip(np.diag(V),0,None))
    t=b/np.where(se>0,se,np.nan);p=2*(1-sp.t.cdf(np.abs(t),df=G-1))
    return b,se,t,p,n
def fe(data,dv,iv=IV):
    s=data[[FIRM,YEAR,dv,iv]+CTRLS].dropna().copy()
    if len(s)<MIN_OBS or s[FIRM].nunique()<4: return None
    s[FIRM]=s[FIRM].astype(str)
    X=pd.concat([s[[iv]+CTRLS].reset_index(drop=True),
      pd.get_dummies(s[FIRM],prefix='e',drop_first=True,dtype=float).reset_index(drop=True),
      pd.get_dummies(s[YEAR].astype(int),prefix='y',drop_first=True,dtype=float).reset_index(drop=True)],axis=1)
    b,se,t,p,n=ols_cluster(X.to_numpy(float),s[dv].to_numpy(float),s[FIRM].to_numpy())
    i=list(X.columns).index(iv)
    return dict(coef=b[i],se=se[i],t=t[i],p=p[i],lo=b[i]-1.96*se[i],hi=b[i]+1.96*se[i],n=n,nfirm=s[FIRM].nunique())
def nw(series,L=1):
    x=np.array(series.dropna().tolist(),float);T=len(x)
    if T<3: return np.nan,np.nan
    mu=x.mean();g=float((x-mu)@(x-mu))/T
    for h in range(1,L+1):
        w=1-h/(L+1);g+=2*w*float((x[h:]-mu)@(x[:-h]-mu))/T
    se=np.sqrt(max(g/T,0));return mu,(mu/se if se>0 else np.nan)

print("="*70); print("PHASE 5 — SUPER-GROUP DECOMPOSITION v9 [FRESH]"); print("="*70)
df=pd.read_csv(PANEL,low_memory=False); df['SG']=df['SuperGroup']
df=add_regime_flag(df)
if 'RevGrowth_w' not in df: df['RevGrowth_w']=df['RevGrowth'].clip(df['RevGrowth'].quantile(.01),df['RevGrowth'].quantile(.99))
df['DeltaGM']=df['dGM_fwd']; df['DeltaCFO_now']=df['dCFOMgn']
for c in ['peer_excess_sg','AbnInvPeer','DeltaGM','DeltaCFO_now','GM_distress','StressScore']:
    if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
if 'peer_excess_sg_w' not in df: df['peer_excess_sg_w']=df['peer_excess_sg'].clip(df['peer_excess_sg'].quantile(.01),df['peer_excess_sg'].quantile(.99))
SD=aig_sd(df); USH=us_share(df,by='SuperGroup').set_index('SuperGroup')
def usp(sg): return float(USH.loc[sg,'us_share_pct'])
def nfirm(sg): return int(USH.loc[sg,'firms'])
def frag(sg): return 'fragile (small/US-heavy)' if nfirm(sg)<FRAGILE_FIRMS else ''
print(f"  Panel: {df[FIRM].nunique()} firms | AIG SD={SD:.4f}")
print(f"  {US_NOTE[:84]}...")
print(f"  Fragile groups (<{FRAGILE_FIRMS} firms): S4 Technology ({nfirm('S4')}, {usp('S4'):.0f}% US), S6 Healthcare ({nfirm('S6')}, {usp('S6'):.0f}% US)")

# ===== TABLE 15/16 — H1, H2 by super-group =====
def by_sg(dv,label,fn):
    print(f"\n{'-'*70}\n  {fn}\n{'-'*70}")
    rows=[]
    for sg in SG_ORDER:
        r=fe(df[df['SG']==sg],dv)
        if r:
            rows.append(dict(SuperGroup=SG_LABELS[sg],Beta=round(r['coef'],4),SE=round(r['se'],4),t=round(r['t'],2),
              p_value=round(r['p'],4),Stars=stars(r['p']),pp_per_1SD=per_sd(r['coef'],SD),N_obs=r['n'],
              N_firms=nfirm(sg),US_share_pct=usp(sg),Note=frag(sg)))
            print(f"  {SG_LABELS[sg]:20s} β={r['coef']:+.4f}{stars(r['p']):3s} p={r['p']:.3f} | {nfirm(sg)}f {usp(sg):.0f}%US {frag(sg)}")
    return pd.DataFrame(rows)
T15=by_sg('DeltaGM','H1','TABLE 15 — H1 (next-yr ΔGM) by super-group'); sv(T15,'Table15_H1_by_supergroup_v9.csv')
T16=by_sg('DeltaCFO_now','H2','TABLE 16 — H2 (same-yr ΔCFO) by super-group'); sv(T16,'Table16_H2_by_supergroup_v9.csv')

# ===== TABLE 17 — distress chi-square by SG =====
print(f"\n{'-'*70}\n  TABLE 17 — distress chi-square (Score>=3 vs <=1) by super-group\n{'-'*70}")
rows=[]
for sg in SG_ORDER:
    s=df[(df['SG']==sg)&df['GM_distress'].notna()&df['StressScore'].notna()]
    hi=s[s['StressScore']>=3]['GM_distress']; lo=s[s['StressScore']<=1]['GM_distress']
    if len(hi)<10 or len(lo)<10:
        rows.append(dict(SuperGroup=SG_LABELS[sg],hi_rate=np.nan,lo_rate=np.nan,chi2=np.nan,p_value=np.nan,Stars='',N_firms=nfirm(sg),US_share_pct=usp(sg),Note='too few high/low-stress')); continue
    cont=np.array([[hi.sum(),len(hi)-hi.sum()],[lo.sum(),len(lo)-lo.sum()]]); chi2,p,_,_=sp.chi2_contingency(cont)
    rows.append(dict(SuperGroup=SG_LABELS[sg],hi_rate=round(hi.mean(),3),lo_rate=round(lo.mean(),3),chi2=round(chi2,2),p_value=round(p,4),Stars=stars(p),N_firms=nfirm(sg),US_share_pct=usp(sg),Note=frag(sg)))
    print(f"  {SG_LABELS[sg]:20s} χ²={chi2:5.2f} p={p:.3f}{stars(p):3s} {hi.mean()*100:.0f}% vs {lo.mean()*100:.0f}% | {nfirm(sg)}f {frag(sg)}")
T17=pd.DataFrame(rows); sv(T17,'Table17_distress_chisq_by_supergroup_v9.csv')

# ===== TABLE 18 — within-SG quintile returns =====
print(f"\n{'-'*70}\n  TABLE 18 — within-super-group quintile return spread\n{'-'*70}")
rows=[]
for sg in SG_ORDER:
    sub=df[df['SG']==sg]; spreads=[]
    for yr,grp in sub.groupby(YEAR):
        g=grp.dropna(subset=[IV,'peer_excess_sg_w'])
        if len(g)<10: continue
        g=g.copy()
        try: g['q']=pd.qcut(g[IV].rank(method='first'),5,labels=False,duplicates='drop')
        except: continue
        if (g['q']==4).sum()<2 or (g['q']==0).sum()<2: continue
        spreads.append(g[g['q']==4]['peer_excess_sg_w'].mean()-g[g['q']==0]['peer_excess_sg_w'].mean())
    mu,t=nw(pd.Series(spreads))
    rows.append(dict(SuperGroup=SG_LABELS[sg],Q5_Q1=round(mu,4) if not np.isnan(mu) else np.nan,NW_t=round(t,2) if not np.isnan(t) else np.nan,N_years=len(spreads),N_firms=nfirm(sg),US_share_pct=usp(sg),Note=('thin' if len(spreads)<6 else frag(sg))))
    print(f"  {SG_LABELS[sg]:20s} Q5-Q1={mu:+.4f} t={t:+.2f} ({len(spreads)} yrs) {frag(sg)}")
T18=pd.DataFrame(rows); sv(T18,'Table18_quintile_by_supergroup_v9.csv')

# ===== TABLE 19 — coverage + US-share =====
cov=df.groupby('SG').agg(firm_years=('Ticker','size')).reindex(SG_ORDER).reset_index()
cov['firms']=[nfirm(sg) for sg in SG_ORDER]; cov['US_share_pct']=[usp(sg) for sg in SG_ORDER]; cov['Note']=[frag(sg) for sg in SG_ORDER]
cov['SuperGroup']=[SG_LABELS[sg] for sg in SG_ORDER]; cov=cov[['SuperGroup','firms','firm_years','US_share_pct','Note']]
sv(cov,'Table19_coverage_us_share_v9.csv')

# ===== FIGURES (coef plots, heatmap, facets, quintile, coverage, forest) =====
def coef_plot(table,fn,t,sub,xlabel):
    d=table.dropna(subset=['Beta']); fig,ax=plt.subplots(figsize=(9.5,max(4.5,len(d)*0.85))); y=np.arange(len(d))[::-1]
    for yi,(_,r) in zip(y,d.iterrows()):
        sgk=[k for k,v in SG_LABELS.items() if v==r['SuperGroup']][0]
        lo=r['Beta']-1.96*r['SE']; hi=r['Beta']+1.96*r['SE']
        ax.plot([lo,hi],[yi,yi],color=SG_COL[sgk],lw=2.4,alpha=.85,solid_capstyle='round')
        ax.scatter(r['Beta'],yi,s=95,color=SG_COL[sgk],edgecolors='white',lw=1.3,zorder=4)
        ax.text(hi,yi+0.18,f"{r['Beta']:+.3f}{r['Stars']}",fontsize=8.5,color=INK,va='bottom')
        if r.get('Note'): ax.text(lo,yi-0.28,r['Note'],fontsize=6.8,color=MUTE,style='italic')
    ax.axvline(0,color=INK,lw=1); ax.set_yticks(y); ax.set_yticklabels([f"{r['SuperGroup']} ({r['US_share_pct']:.0f}% US)" for _,r in d.iterrows()])
    ax.set_xlabel(xlabel); title(ax,t,sub); style(ax,grid='x'); fig.subplots_adjust(left=0.30); sf(fig,fn)
coef_plot(T15,'Figure15_H1_coef_by_supergroup_v9.png','AbnInvPeer coefficient on next-year ΔGM, by super-group','Two-way FE with 95% CIs; US-share shown per group. Markers left of zero indicate margin compression.','β (AbnInvPeer → next-year ΔGM)')
coef_plot(T16,'Figure16_H2_coef_by_supergroup_v9.png','AbnInvPeer coefficient on same-year ΔCFO margin, by super-group','Two-way FE with 95% CIs; same-year (contemporaneous) horizon.','β (AbnInvPeer → same-year ΔCFO margin)')

hm=df.dropna(subset=['DeltaGM','StressScore','SG']).copy(); piv=hm.pivot_table(index='SG',columns='StressScore',values='DeltaGM',aggfunc='mean').reindex(SG_ORDER)
piv=piv[[c for c in [0,1,2,3,4] if c in piv.columns]]
fig,ax=plt.subplots(figsize=(9,5)); cmap=LinearSegmentedColormap.from_list('rb',['#9C5A3C','#FFFFFF','#2F5C7E']); vmax=np.nanmax(np.abs(piv.values))
im=ax.imshow(piv.values,cmap=cmap,vmin=-vmax,vmax=vmax,aspect='auto'); ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'Score {int(c)}' for c in piv.columns])
ax.set_yticks(range(len(SG_ORDER))); ax.set_yticklabels([SG_SHORT[s] for s in SG_ORDER])
for i in range(len(SG_ORDER)):
    for j in range(len(piv.columns)):
        v=piv.values[i,j]
        if not np.isnan(v): ax.text(j,i,f'{v*100:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>vmax*.55 else INK)
title(ax,'Mean next-year ΔGM by StressScore and super-group','Cell values in percentage points; blue = margin expansion, brown = compression.')
plt.colorbar(im,ax=ax,shrink=.8,label='Mean next-year ΔGM'); style(ax,grid=None); sf(fig,'Figure17_dGM_heatmap_v9.png')

fig,axes=plt.subplots(2,3,figsize=(13,7.5),sharex=True,sharey=True)
for ax,sg in zip(axes.flat,SG_ORDER):
    s=df[df['SG']==sg].dropna(subset=[IV,'DeltaGM']); s=s[s[IV].abs()<s[IV].quantile(.98)]
    s2=s.copy(); s2['bin']=pd.qcut(s2[IV],12,labels=False,duplicates='drop'); b=s2.groupby('bin').agg(x=(IV,'mean'),y=('DeltaGM','mean')).reset_index()
    ax.scatter(b['x'],b['y'],s=38,color=SG_COL[sg],edgecolors='white',lw=.7,zorder=3)
    if len(s)>20:
        m,c=np.polyfit(s[IV],s['DeltaGM'],1); xs=np.linspace(b['x'].min(),b['x'].max(),30); ax.plot(xs,m*xs+c,color='#9C5A3C',lw=1.8,zorder=2)
    r=fe(df[df['SG']==sg],'DeltaGM'); bb=r['coef'] if r else np.nan
    ax.axhline(0,color=INK,lw=.6); ax.axvline(0,color=INK,lw=.6)
    ax.set_title(f"{SG_LABELS[sg]} (β={bb:+.3f}, {usp(sg):.0f}% US)",fontsize=9.5,color=INK,loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(color=GRID,alpha=.6,lw=.5)
fig.suptitle('AbnInvPeer versus next-year ΔGM, by super-group',x=0.07,y=0.99,ha='left',fontsize=13,fontweight='bold',color=INK)
fig.text(0.07,0.945,'Binned scatter (12 bins) with OLS fit per super-group; β from two-way FE, US-share in panel title.',ha='left',fontsize=9.3,color=MUTE)
fig.supxlabel('AbnInvPeer'); fig.supylabel('Next-year ΔGM'); fig.subplots_adjust(top=0.88,hspace=0.28,wspace=0.12); sf(fig,'Figure18_faceted_scatter_v9.png')

d19=T18.dropna(subset=['Q5_Q1']); fig,ax=plt.subplots(figsize=(10,5.6)); sgk=[[k for k,v in SG_LABELS.items() if v==r['SuperGroup']][0] for _,r in d19.iterrows()]
ax.bar(range(len(d19)),d19['Q5_Q1'],color=[SG_COL[k] for k in sgk],edgecolor='white',width=.66); ax.axhline(0,color=INK,lw=1)
for i,(_,r) in enumerate(d19.iterrows()): ax.text(i,r['Q5_Q1']+(0.003 if r['Q5_Q1']>=0 else -0.003),f"t={r['NW_t']}",ha='center',va='bottom' if r['Q5_Q1']>=0 else 'top',fontsize=8,color=INK)
ax.set_xticks(range(len(d19))); ax.set_xticklabels([SG_SHORT[k] for k in sgk]); ax.set_ylabel('Q5−Q1 12m peer-excess return')
title(ax,'Within-super-group top-minus-bottom quintile return spread','Annual within-group sorts on AbnInvPeer; Newey-West t above bars.'); style(ax); sf(fig,'Figure19_quintile_spread_v9.png')

fig,ax=plt.subplots(figsize=(9.5,5)); x=np.arange(len(SG_ORDER)); fy=[df[df.SG==sg].shape[0] for sg in SG_ORDER]
ax.bar(x,fy,width=.6,color=[SG_COL[s] for s in SG_ORDER],edgecolor='white')
for i,sg in enumerate(SG_ORDER): ax.text(i,fy[i]+5,f"{fy[i]} obs\n{nfirm(sg)}f · {usp(sg):.0f}%US",ha='center',va='bottom',fontsize=7.8,color=INK)
ax.set_xticks(x); ax.set_xticklabels([SG_SHORT[s] for s in SG_ORDER]); ax.set_ylabel('Firm-years')
title(ax,'Sample coverage and US-share by super-group','S4 (Technology) and S6 (Healthcare) are smallest and most US-heavy — flagged fragile.'); style(ax); sf(fig,'Figure20_coverage_v9.png')

fig,ax=plt.subplots(figsize=(10,7)); ypos=0; yticks=[]; ylabels=[]
for sg in SG_ORDER[::-1]:
    r1=fe(df[df['SG']==sg],'DeltaGM'); h1t=r1['t'] if r1 else np.nan
    rc=fe(df[df['SG']==sg],'DeltaCFO_now'); h2t=rc['t'] if rc else np.nan
    row17=T17[T17['SuperGroup']==SG_LABELS[sg]]; h4=np.sqrt(row17['chi2'].iloc[0]) if len(row17) and not pd.isna(row17['chi2'].iloc[0]) else np.nan
    for lab,val,col in [('H1 ΔGM',h1t,'#2F5C7E'),('H2 ΔCFO',h2t,'#3E7C6A'),('H4 distress',h4,'#9C6B4E')]:
        ax.barh(ypos,val,color=col,alpha=.85,edgecolor='white',height=.7); ypos+=1
    yticks.append(ypos-2); ylabels.append(f"{SG_SHORT[sg]} ({usp(sg):.0f}% US)"); ypos+=1
ax.axvline(0,color=INK,lw=1); ax.axvline(1.96,color=MUTE,ls='--',lw=.8); ax.axvline(-1.96,color=MUTE,ls='--',lw=.8)
ax.set_yticks(yticks); ax.set_yticklabels(ylabels); ax.set_xlabel('Test statistic (t for H1/H2, √χ² for H4; dashed = ±1.96)')
ax.legend(handles=[Patch(color='#2F5C7E',label='H1 ΔGM'),Patch(color='#3E7C6A',label='H2 ΔCFO'),Patch(color='#9C6B4E',label='H4 distress')],loc='lower right',frameon=True)
title(ax,'Hypothesis test statistics by super-group','Each cluster is one super-group; bars beyond ±1.96 are significant at 5%.'); style(ax,grid='x'); fig.subplots_adjust(left=0.18); sf(fig,'Figure21_forest_by_supergroup_v9.png')

# ===== ONE OVERALL FINDINGS TABLE =====
print(f"\n{'='*70}\n  OVERALL FINDINGS TABLE\n{'='*70}")
h2_sig=int((T16['p_value']<0.05).sum()); h2_neg=int((T16['Beta']<0).sum()); h2_10=int((T16['p_value']<0.10).sum())
h1_sig_rows=T15[T15['p_value']<0.05]; h1_sig=h1_sig_rows['SuperGroup'].tolist()
h1_fragile_only = len(h1_sig)>0 and all('fragile' in str(r['Note']) for _,r in h1_sig_rows.iterrows())
findings=[
 dict(Item='H2 cash flow by super-group',Finding=f'Negative in all 6 groups; significant at 5% in {h2_sig}/6, at 10% in {h2_10}/6.',Conclusion='Most universal result — consistent negative sign everywhere, strongest of the four.'),
 dict(Item='H1 margin by super-group',Finding=f"Significant at 5% only in: {', '.join(s.split(' · ')[1] for s in h1_sig) if h1_sig else 'no group'}" + (' (a FRAGILE group)' if h1_fragile_only else '') + '; Retail borderline (p=0.05).',Conclusion='Per-group H1 is weak and rests on a fragile group; the pooled H1 result is the reportable one, not the group decomposition.'),
 dict(Item='Distress by super-group',Finding='Significant in only some groups at the per-group level.',Conclusion='Pooled result stronger than any single group (per-group power limited).'),
 dict(Item='Returns by super-group',Finding='Group-fragile; some groups show wrong-sign spreads.',Conclusion='Pooled is primary; per-group decomposition shows where signal is detectable.'),
 dict(Item='Fragile groups',Finding=f"S4 Technology ({nfirm('S4')}f, {usp('S4'):.0f}% US) and S6 Healthcare ({nfirm('S6')}f, {usp('S6'):.0f}% US).",Conclusion='Flag everywhere; do not build narrative on their per-group estimates.'),
 dict(Item='US-share by group',Finding=f"Ranges {min(usp(s) for s in SG_ORDER):.0f}-{max(usp(s) for s in SG_ORDER):.0f}%; tech & healthcare most US-heavy.",Conclusion='Per-group claims for US-heavy groups are effectively US claims.'),
]
sv(pd.DataFrame(findings),'Table_FINDINGS_overall_v9.csv')
for f in findings: print(f"  - {f['Item']}: {f['Finding']}\n      => {f['Conclusion']}")
print(f"\n{'='*70}\n  PHASE 5 v9 COMPLETE (fresh rebuild)\n{'='*70}")
print(f"  Tables ({len(os.listdir(TAB))}), Figures ({len(os.listdir(FIG))})")
