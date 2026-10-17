"""2560x1440 3D scene and instrument display, all telemetry from simulation states."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter
from matplotlib.patches import Arc,Circle,FancyBboxPatch,Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from simulate import INERTIA,ROOT,OBSTACLES,LOC,DIR,FACES,box_vertices,MASS,GRAVITY,ROLL_START,ROLL_END
BG='#070b10';DIM='#52606d';WHITE='#eef3f6';CYAN='#67deff'

def wheel_mesh(x,y):
 a=np.linspace(0,2*np.pi,13);rings=[np.column_stack([x+.25*np.cos(a),np.full(len(a),yy),-.3+.25*np.sin(a)]) for yy in [y-.08,y+.08]]
 return [rings[0],rings[1]]+[np.array([rings[0][i],rings[0][i+1],rings[1][i+1],rings[1][i]]) for i in range(12)]

def nozzle_mesh(origin,direction):
 axis=direction;u=np.cross(axis,[1.,0,0])
 if np.linalg.norm(u)<.1:u=np.cross(axis,[0.,1,0])
 u/=np.linalg.norm(u);v=np.cross(axis,u);angles=np.linspace(0,2*np.pi,9)
 rings=[origin+axis*length+radius*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v) for length,radius in [(0,.035),(.13,.085)]]
 return [np.array([rings[0][i],rings[0][i+1],rings[1][i+1],rings[1][i]]) for i in range(8)]+[rings[1]],rings[1].mean(axis=0)

BODY=box_vertices(np.array([-2.25,-1.,-.3]),np.array([2.25,1.,.15]));CAB=box_vertices(np.array([-.7,-.75,.15]),np.array([.7,.75,.65]))
PARTS=[([BODY[f] for f in FACES],'#e8eef4','#99afbf'),([CAB[f] for f in FACES],'#46677d','#a7c4d6')]
for x in [-1.3,1.3]:
 for y in [-.9,.9]:PARTS.append((wheel_mesh(x,y),'#101820','#8395a4'))
NOZZLES=[nozzle_mesh(LOC[j],-DIR[j]) for j in range(10)]

def gauge(fig,rect,label,unit,maximum):
 ax=fig.add_axes(rect,facecolor=BG);ax.set(xlim=(-1.2,1.2),ylim=(-1.15,1.15));ax.set_aspect('equal');ax.axis('off')
 ax.add_patch(Arc((0,0),2,2,theta1=-40,theta2=220,color='#333b43',lw=3))
 fill=Arc((0,0),2,2,theta1=220,theta2=220,color=WHITE,lw=3);ax.add_patch(fill)
 ax.text(0,.38,label,ha='center',color='#a6b2bc',fontsize=12)
 value=ax.text(0,-.03,'0.0',ha='center',va='center',color=WHITE,fontsize=35,fontfamily='DejaVu Sans Mono')
 ax.text(0,-.46,unit,ha='center',color='#a6b2bc',fontsize=11)
 def update(x):
  value.set_text(f'{x:.1f}');fill.theta1=220-260*np.clip(x/maximum,0,1);fill.theta2=220
 return update

def render(A,preview_only=False):
 fig=plt.figure(figsize=(25.6,14.4),dpi=100,facecolor=BG)
 ax=fig.add_axes([.005,.235,.99,.66],projection='3d',facecolor=BG,computed_zorder=False)
 scene_end=max(36.,float(A[:,1].max())+4.)
 ax.set(xlim=(-3,scene_end),ylim=(-5,5),zlim=(-.1,9));ax.set_box_aspect([scene_end+3,10,9],zoom=1.65);ax.set_proj_type('ortho');ax.view_init(elev=18,azim=70);ax.set_axis_off()
 for x in np.arange(-3,scene_end+1,3):ax.plot([x,x],[-5,5],[0,0],color='#202b36',lw=.6,zorder=1,clip_on=False)
 for y in np.arange(-5,6,2):ax.plot([-3,scene_end],[y,y],[0,0],color='#202b36',lw=.6,zorder=1,clip_on=False)
 ax.plot([9.7,23.],[3.8,3.8],[.02,.02],color='#df3e46',lw=3,zorder=3,clip_on=False)
 for ob in OBSTACLES:
  lo=np.array(ob['lo']);hi=np.array(ob['hi']);vs=box_vertices(lo,hi)
  ax.add_collection3d(Poly3DCollection([vs[f] for f in FACES],facecolors=ob['color'],edgecolors='#98a9b8',linewidths=.6,zorder=2))
  x1,x2=lo[0]+.15,hi[0]-.15;y=hi[1]+.015;h=hi[2]
  glass=np.array([[x1,y,h*.57],[x2,y,h*.57],[x2,y,h*.85],[x1,y,h*.85]])
  ax.add_collection3d(Poly3DCollection([glass],facecolors='#112532',edgecolors='#88a5b8',linewidths=.6,zorder=3))
  for x in [x1+.1,x2-.1]:ax.plot([x-.1,x+.1],[y,y],[h*.29,h*.29],color='#d1f0ff',lw=2,zorder=3,clip_on=False)
  ax.text((x1+x2)/2,3.9,-.45,ob['name'].upper(),ha='center',color='#8b9dab',fontsize=10,zorder=3,clip_on=False)
 trail,=ax.plot([],[],[],color='#67deff',lw=1.6,alpha=.9,zorder=6,clip_on=False)
 jet_lines=[ax.plot([],[],[],color=CYAN,lw=3,alpha=.8,zorder=13,clip_on=False)[0] for _ in range(10)]
 main_artists=[]
 fig.text(.035,.945,'ROADSTER  /  FLIGHT TEST',color=WHITE,fontsize=24,weight='bold')
 fig.text(.035,.913,'3D RIGID-BODY SIMULATION     •     PID + FEEDFORWARD     •     UNLIMITED GAS',color='#778d9e',fontsize=11)
 phase=fig.text(.965,.945,'BOOST',ha='right',color=CYAN,fontsize=20,fontfamily='DejaVu Sans Mono')
 clock=fig.text(.965,.914,'',ha='right',color='#9db0bf',fontsize=12)
 # Instruments inspired by the supplied screenshots. Values come from actual states.
 update_g=gauge(fig,[.025,.016,.155,.205],'G-FORCE','G',3.)
 update_h=gauge(fig,[.815,.016,.155,.205],'ALTITUDE','FT',30.)
 h_m=fig.text(.892,.019,'',ha='center',color='#899ba9',fontsize=10)
 att=fig.add_axes([.20,.015,.18,.21],projection='3d',facecolor=BG,computed_zorder=False)
 att.set(xlim=(-3,3),ylim=(-3,3),zlim=(-3,3));att.set_box_aspect([1,1,1]);att.set_proj_type('ortho');att.view_init(elev=18,azim=70);att.set_axis_off()
 a=np.linspace(0,2*np.pi,80)
 for pts in [np.array([3*np.cos(a),3*np.sin(a),a*0]),np.array([3*np.cos(a),a*0,3*np.sin(a)]),np.array([a*0,3*np.cos(a),3*np.sin(a)])]:att.plot(*pts,color='#3b4650',lw=.7,alpha=.8,zorder=1,clip_on=False)
 fig.text(.29,.215,'ATTITUDE',ha='center',color='#9eabb6',fontsize=11)
 att_artists=[]
 # Central flight-state readout.
 state_ax=fig.add_axes([.40,.025,.17,.185],facecolor=BG);state_ax.axis('off')
 state_ax.text(0,1,'FLIGHT STATE',color='#9eabb6',fontsize=12)
 state_text=state_ax.text(0,.81,'',va='top',color=WHITE,fontsize=15,fontfamily='DejaVu Sans Mono',linespacing=1.7)
 # All nozzles stay visible; cyan indicates measured actual thrust above 30 N.
 panel=fig.add_axes([.60,.012,.17,.205],facecolor=BG);panel.set(xlim=(-2.7,2.7),ylim=(-3.1,3.1));panel.set_aspect('equal');panel.axis('off')
 panel.add_patch(FancyBboxPatch((-1,-2.2),2,4.4,boxstyle='round,pad=.08,rounding_size=.45',fc='#141d26',ec='#758b9d',lw=1.1))
 panel.add_patch(Polygon([[-.7,.5],[-.7,1.35],[.7,1.35],[.7,.5]],fc='#233849',ec='#627f94',lw=.6))
 markers=[]
 for j,loc in enumerate(LOC):
  xx=-loc[1];yy=loc[0]
  if j in [4,5]:xx*=1.15
  marker=Circle((xx,yy),.16 if j<6 else .09,fc='#253340',ec='#8094a3',lw=1.2);panel.add_patch(marker);markers.append(marker)
 panel.text(0,2.72,'THRUSTER STATUS',ha='center',color='#9eabb6',fontsize=11)
 panel.text(-1.7,2.35,'FL',ha='center',color='#9eabb6',fontsize=9);panel.text(1.7,2.35,'FR',ha='center',color='#9eabb6',fontsize=9)
 side_beams=[]
 for sign in [-1,1]:
  beam=Polygon([[sign*1.23,1.9],[sign*2.4,2.12],[sign*2.4,1.68]],fc=CYAN,ec='none',alpha=0.);panel.add_patch(beam);side_beams.append(beam)
 side_text=panel.text(0,-2.95,'',ha='center',color='#b9cbd8',fontsize=9,fontfamily='DejaVu Sans Mono')
 footer=fig.text(.5,.004,'ASSUMED VEHICLE PARAMETERS  •  INITIAL YAW ERROR +2.5°  •  ENDS AT FIRST GROUND CONTACT',ha='center',color='#526574',fontsize=9)
 import json
 descent_start=json.loads((ROOT/'metrics.json').read_text())['final_descent_start_s']
 # Reserve the right 36% for six scientific plots, retaining the full 3D view/HUD.
 for axis in list(fig.axes):
  pos=axis.get_position(original=True);axis.set_position([pos.x0*.61,pos.y0,pos.width*.61,pos.height])
 for text in fig.texts:
  xx,yy=text.get_position();text.set_position((xx*.61,yy))
 from analysis_panels import add_live_panels,static_plots
 update_panels=add_live_panels(fig,A)
 from display_reference import display_reference
 ref,guide=display_reference(A)
 np.savetxt(ROOT/'display_reference.csv',np.column_stack([A[:,0],ref,guide.astype(int)]),delimiter=',',header='time_s,x_aim_m,y_aim_m,z_aim_m,is_interpolated_roll_guide',comments='')
 ax.plot(ref[:,0],ref[:,1],ref[:,2],'--',color='#ffd166',lw=2.,zorder=7,clip_on=False)
 fig.text(.025,.875,'TARGET / SMOOTH ROLL GUIDE',color='#ffd166',fontsize=10)
 fig.text(.27,.875,'ACTUAL PID/PD MOTION',color=CYAN,fontsize=10)
 # Parameter panel occupies the open space above the flight scene.
 param=fig.add_axes([.025,.69,.57,.145],facecolor='#0b1219');param.set_xticks([]);param.set_yticks([])
 for spine in param.spines.values():spine.set_color('#2d404f')
 param.text(.02,.86,'VEHICLE PARAMETERS  /  ASSUMED MODEL',color=WHITE,fontsize=12,weight='bold')
 columns=[
  (0.02,[f'Mass: {MASS:,.0f} kg',f'Ixx / roll: {INERTIA[0]:,.0f} kg·m²',f'Iyy / pitch: {INERTIA[1]:,.0f} kg·m²',f'Izz / yaw: {INERTIA[2]:,.0f} kg·m²']),
  (.35,['Envelope: 4.5 × 2.3 × 1.2 m','COM: 0.55 m above base','Body origin: centre of mass','Inertia: diagonal, body axes']),
  (.69,['Underside: 4 × 10 kN max','Front sides: 2 × 2 kN max','Rear: 4 nozzles, inactive','Valve lag: 30 ms; unlimited gas'])]
 for x,lines in columns:
  for y,line in zip([.64,.47,.30,.13],lines):param.text(x,y,line,color='#b9cbd8',fontsize=10.5)
 fig.text(.025,.854,'Roll section: smooth interpolated guide; position PID is off',color='#95adbf',fontsize=9)
 angles=Rotation.from_quat(A[:,7:11]).as_euler('xyz',degrees=True);roll=np.rad2deg(np.unwrap(np.deg2rad(angles[:,0])))
 def draw_vehicle(axis,storage,p,R,forces,with_nozzles):
  for obj in storage:obj.remove()
  storage.clear()
  for faces,color,edge in PARTS:
   obj=Poly3DCollection([p+v@R.T for v in faces],facecolors=color,edgecolors=edge,linewidths=.55,zorder=10,clip_on=False);axis.add_collection3d(obj);storage.append(obj)
  if with_nozzles:
   faces=[];colors=[]
   for j,(mesh,mouth) in enumerate(NOZZLES):
    faces.extend(p+v@R.T for v in mesh);colors.extend([('#71dfff' if forces[j]>30 else '#8796a1')]*8+['#152936'])
   obj=Poly3DCollection(faces,facecolors=colors,edgecolors='#a9c1cd',linewidths=.5,zorder=12,clip_on=False);axis.add_collection3d(obj);storage.append(obj)
  # Orange front light, so heading remains unambiguous through the roll.
  pts=np.array([p+R@np.array([2.26,-.75,.1]),p+R@np.array([2.26,.75,.1])]);line=axis.plot(*pts.T,color='#ffa85e',lw=2.2,zorder=14,clip_on=False)[0];storage.append(line)
 def update(i):
  row=A[i];p=row[1:4];R=Rotation.from_quat(row[7:11]).as_matrix();f=row[14:24]
  draw_vehicle(ax,main_artists,p,R,f,True);draw_vehicle(att,att_artists,np.zeros(3),R,f,False)
  trail.set_data_3d(A[:i+1,1],A[:i+1,2],A[:i+1,3])
  for j,line in enumerate(jet_lines):
   mouth=p+R@NOZZLES[j][1]
   if f[j]<30:line.set_data_3d([],[],[]);continue
   length=(.6+3.0*f[j]/2000) if j in [4,5] else 1.7*f[j]/10000
   end=mouth-R@DIR[j]*length;line.set_data_3d(np.array([mouth,end]).T)
  proper_g=np.linalg.norm(DIR.T@f)/(MASS*GRAVITY);update_g(proper_g);update_h(p[2]/.3048);h_m.set_text(f'{p[2]:.2f} m')
  label='BOOST' if row[0]<ROLL_START else 'ROLL' if row[0]<ROLL_END else 'BRAKE / DESCEND'
  if row[0]>=descent_start:label='SMOOTH LANDING'
  if i==len(A)-1:label='FIRST CONTACT'
  phase.set_text(label);clock.set_text(f'T + {row[0]:05.2f} s    |    {np.linalg.norm(row[4:7])*3.6:05.1f} km/h')
  state_text.set_text(f'ROLL  {roll[i]:6.1f}°\nPITCH {angles[i,1]:9.5f}°\nYAW   {angles[i,2]:6.2f}°\nVz    {row[6]:6.2f} m/s')
  for j,m in enumerate(markers):m.set_facecolor(CYAN if f[j]>30 else '#253340');m.set_edgecolor('#eefaff' if f[j]>30 else '#8094a3')
  for beam,j in zip(side_beams,[4,5]):beam.set_alpha(.25+.5*min(f[j]/2000,1.) if f[j]>30 else 0.)
  side_text.set_text(f'FL {f[4]:4.0f} N  FR {f[5]:4.0f} N')
  update_panels(i)
  return main_artists
 ids=list(range(0,len(A),round(.05/(A[1,0]-A[0,0]))))+[len(A)-1]
 if preview_only:
  update(0);fig.savefig(ROOT/'hud_launch.png',dpi=100)
  update(int(np.argmin(abs(A[:,0]-1.75))));fig.savefig(ROOT/'hud_roll.png',dpi=100)
  update(len(A)-1);fig.savefig(ROOT/'hud_contact.png',dpi=100)
  plt.close(fig);return
 ani=FuncAnimation(fig,update,frames=ids,interval=50,blit=False)
 ani.save(ROOT/'roadster_smooth_landing.mp4',writer=FFMpegWriter(fps=20,codec='libx264',bitrate=8000,extra_args=['-pix_fmt','yuv420p','-threads','2']),dpi=100)
 import subprocess
 # Streaming two-pass GIF conversion avoids retaining all QHD frames in memory.
 palette=ROOT/'palette.png'
 subprocess.run(['ffmpeg','-y','-v','error','-i',str(ROOT/'roadster_smooth_landing.mp4'),'-vf','fps=15,scale=1792:-1:flags=lanczos,palettegen', '-frames:v','1',str(palette)],check=True)
 subprocess.run(['ffmpeg','-y','-v','error','-i',str(ROOT/'roadster_smooth_landing.mp4'),'-i',str(palette),'-lavfi','fps=15,scale=1792:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer','-loop','0',str(ROOT/'roadster_smooth_landing.gif')],check=True)
 palette.unlink(missing_ok=True)
 static_plots(A)
 update(0);fig.savefig(ROOT/'hud_launch.png',dpi=100)
 update(int(np.argmin(abs(A[:,0]-1.75))));fig.savefig(ROOT/'hud_roll.png',dpi=100)
 update(len(A)-1);fig.savefig(ROOT/'hud_contact.png',dpi=100)
 plt.close(fig)
