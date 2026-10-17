"""Animated scientific plots of actual states/forces and actual logged commands."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation
from simulate import ROOT,ROLL_START,ROLL_END,OBSTACLES
from display_reference import display_reference
BG='#070b10';ACTUAL='#67deff';TARGET='#ffd166';TEXT='#c0cbd4'

def data(A):
 t=A[:,0];ea=Rotation.from_quat(A[:,7:11]).as_euler('xyz',degrees=True)
 ed=Rotation.from_quat(A[:,30:34]).as_euler('xyz',degrees=True)
 roll=np.rad2deg(np.unwrap(np.deg2rad(ea[:,0])));roll_d=np.rad2deg(np.unwrap(np.deg2rad(ed[:,0])))
 ref,_=display_reference(A)
 return t,ea,ed,roll,roll_d,ref

def style(ax,title,xlab,ylab):
 ax.set_facecolor('#0b1219');ax.set_title(title,color='white',loc='left',fontsize=11,pad=9,weight='bold')
 ax.set_xlabel(xlab,color=TEXT,fontsize=9);ax.set_ylabel(ylab,color=TEXT,fontsize=9)
 ax.tick_params(colors='#94a8b8',labelsize=9,length=2)
 for spine in ax.spines.values():spine.set_color('#32414f')
 ax.grid(alpha=.14,color='#8ba1b4')

def populate(fig,A,axes):
 t,ea,ed,roll,roll_d,ref=data(A);T=t[-1]
 specs=[('Trajectory comparison','x (m)','z (m)'),('Height comparison','Time (s)','Height (m)'),('Four underside thrusters','Time (s)','Actual thrust (kN)'),('Two front lateral thrusters','Time (s)','Actual thrust (kN)'),('Flip angle / roll','Time (s)','Unwrapped angle (deg)'),('Pitch: target vs actual','Time (s)','Pitch (deg)')]
 for ax,spec in zip(axes,specs):style(ax,*spec)
 ax=axes[0]
 for ob in OBSTACLES:ax.add_patch(Rectangle((ob['lo'][0],0),ob['hi'][0]-ob['lo'][0],ob['hi'][2],fc=ob['color'],alpha=.3))
 ax.plot(ref[:,0],ref[:,2],'--',color=TARGET,lw=1.8,label='Target / roll guide',zorder=5)
 trajectory,=ax.plot([],[],color=ACTUAL,lw=1.6,label='PID/PD actual');dot,=ax.plot([],[],'o',color='white',ms=3)
 ax.set(xlim=(max(34.,float(A[:,1].max())+3.),-2),ylim=(0,8));ax.legend(fontsize=8,facecolor=BG,labelcolor=TEXT,edgecolor='none',loc='upper right')
 sets=[];cursors=[]
 for ax in axes[1:]:
  ax.set_xlim(0,T);ax.axvspan(ROLL_START,ROLL_END,color='#8997a6',alpha=.10)
  cursors.append(ax.axvline(0,color='white',alpha=.45,lw=.7))
 axes[1].plot(t,ref[:,2],'--',color=TARGET,lw=1.8,label='Target / roll guide',zorder=5)
 line,=axes[1].plot([],[],color=ACTUAL,lw=1.4,label='Actual');sets.append((line,A[:,3]))
 axes[1].set_ylim(0,8)
 for j,(name,color) in enumerate(zip(['Front left','Front right','Rear left','Rear right'],['#67deff','#ffb454','#95df7b','#d4a1ff'])):
  line,=axes[2].plot([],[],color=color,lw=1.3,label=name);sets.append((line,A[:,14+j]/1000))
 axes[2].set_ylim(-.15,10.5)
 for j,(name,color) in enumerate(zip(['Front left','Front right'],['#ff80b5','#c3ea76'])):
  line,=axes[3].plot([],[],color=color,lw=1.4,label=name);sets.append((line,A[:,18+j]/1000))
 axes[3].set_ylim(-.05,2.05)
 axes[4].plot(t,roll_d,'--',color=TARGET,lw=1.3,label='Angle target')
 line,=axes[4].plot([],[],color=ACTUAL,lw=1.5,label='Actual');sets.append((line,roll));axes[4].set_ylim(-15,380);axes[4].set_yticks([0,90,180,270,360])
 axes[5].plot(t,ed[:,1],'--',color=TARGET,lw=1.3,label='Pitch target')
 line,=axes[5].plot([],[],color=ACTUAL,lw=1.5,label='Actual');sets.append((line,ea[:,1]));pitch_extent=max(.005,1.2*float(np.max(np.abs(np.r_[ea[:,1],ed[:,1]]))))
 axes[5].set_ylim(-pitch_extent,pitch_extent)
 from matplotlib.ticker import FormatStrFormatter
 axes[5].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
 axes[5].text(.02,.87,f'Actual range: {ea[:,1].min():.5f}° to {ea[:,1].max():.5f}°',transform=axes[5].transAxes,color='#9dd5e5',fontsize=7.5)
 # Mark the continuous forward landing arc.
 import json
 metrics=json.loads((ROOT/'metrics.json').read_text());start=metrics['final_descent_start_s']
 axes[5].text(.02,.76,f"Below 2 m: max |pitch| {metrics['maximum_absolute_pitch_below_2m_deg']:.5f}°",transform=axes[5].transAxes,color='#9dd5e5',fontsize=7.5)
 if start is not None:
  axes[5].axvspan(start,T,color='#4dbb99',alpha=.10);axes[5].text(.98,.05,'Green: smooth landing',transform=axes[5].transAxes,color='#8bcbb7',ha='right',fontsize=8)
 for k,ax in enumerate(axes[1:],1):ax.legend(fontsize=8,ncol=2 if k==2 else 1,facecolor=BG,labelcolor=TEXT,edgecolor='none',loc='upper right')
 axes[1].text(.02,.03,'Roll: smooth guide; position PID off',transform=axes[1].transAxes,color='#91a2b1',fontsize=7.5)
 def update(i):
  trajectory.set_data(A[:i+1,1],A[:i+1,3]);dot.set_data([A[i,1]],[A[i,3]])
  for line,values in sets:line.set_data(t[:i+1],values[:i+1])
  for cursor in cursors:cursor.set_xdata([t[i],t[i]])
 return update

def add_live_panels(fig,A):
 axes=[]
 for row in range(3):
  for col in range(2):axes.append(fig.add_axes([.655+.18*col,.69-.30*row,.15,.235]))
 fig.text(.655,.98,'LIVE CONTROL ANALYSIS',color='white',fontsize=17,weight='bold')
 fig.text(.655,.955,'Dashed = target / roll guide   •   Solid = actual   •   Grey = roll',color='#95adbf',fontsize=10)
 return populate(fig,A,axes)

def static_plots(A):
 fig,axes=plt.subplots(3,2,figsize=(16,12),facecolor=BG)
 fig.subplots_adjust(left=.08,right=.98,top=.92,bottom=.07,hspace=.36,wspace=.24)
 fig.suptitle('PID baseline — target, motion and actual nozzle forces',color='white',fontsize=18)
 update=populate(fig,A,list(axes.flat));update(len(A)-1)
 fig.savefig(ROOT/'control_analysis.png',dpi=150,facecolor=BG);plt.close(fig)
