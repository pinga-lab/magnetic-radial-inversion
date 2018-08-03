# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 11:26:10 2017

@author: felipemelo
"""
import numpy as np
import time

#casa
arquivo='C:\Users\Felipe\Coisas-Doutorado-sem-backup\DADOS_REAIS\Goias\AREA_I\GDB_XYZ\Mag.XYZ'

lat1=[]
long1=[]
y=[]
x=[]
d=[]
igrf1=[]

with open(arquivo) as infile:
	for line in infile:
		if line[:4]=='Line':
			pass
		elif line[:3]=='/ -':
			pass
		elif line[135:136]=='*':
			pass	
#		elif (line[63:66]=='-19' or line[63:66]=='-20') and (line[77:80]=='-51' or line[77:80]=='-52' or line[77:80]=='-53'):
		elif (line[149:152]=='-51') and (line[161:164]=='-16'):
			x.append(line[2:11])
			y.append(line[13:23])
			lat1.append(line[161:171])
			long1.append(line[149:159])
			d.append(line[127:136])
			igrf1.append(line[138:147])
		else:
			pass
         
lat=filter(None,lat1)
long=filter(None,long1)			
ycord=filter(None,y)
xcord=filter(None,x)
data=filter(None,d)
igrf=filter(None,igrf1)
         
out=np.array([xcord,ycord,long,lat,data,igrf])        
out=out.T.astype(np.float)

#np.save('mag_y_x_tmi',out)

np.savetxt('mag_y_x_tmi.txt',out,delimiter=' ',fmt='%1.8f')

