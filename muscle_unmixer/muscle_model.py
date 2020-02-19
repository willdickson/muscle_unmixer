#!/usr/bin/env python

##Library for aligning muscle models to the thorax of the fly
import os
import copy
import cPickle
import numpy as np


class Frame(dict):    
    def __setitem__(self,key,item):
        try:
            if key in ['a1','a2']:
                dict.__setitem__(self,key,item)
                A = np.vstack((self['a1'],self['a2'])).T
                A_inv = np.linalg.inv(A)
                self['A'] = A
                self['A_inv'] = A_inv
            else:
                dict.__setitem__(self,key,item)
        except KeyError:
            dict.__setitem__(self,key,item)

    def load(self,filepath):
        """load from a pickled dictionary"""
        pkname = filepath
        f = open(pkname);data = cPickle.load(f);f.close()
        for key in data.keys():
            self[key] = data[key]

    def get_transform(self,other):
        """get transform into self from other frame"""
        A1 = np.hstack((self['A'],self['p'][:,np.newaxis]))
        A2 = np.hstack((other['A'],other['p'][:,np.newaxis]))
        A1 = np.vstack((A1,[0,0,1]))
        A2 = np.vstack((A2,[0,0,1]))
        return(np.dot(A1,np.linalg.inv(A2)))


class GeometricModel(object):   
    def __init__(self,lines = None,frame = None,filepath = None):
        if not(filepath is None):
            self.load(filepath)
        else:
            self.construct(lines,frame)

    def construct(self,lines,frame):
        self.lines = lines
        self.frame = frame
        ## put lines in barycentric coords
        self.barycentric = dict()
        for key in self.lines.keys():
            coords = np.dot(self.frame['A_inv'],(self.lines[key]-self.frame['p'][:,np.newaxis])) 
            self.barycentric[key] = coords.T
            
    def load(self,filepath):
        """load from a pickled dictionary"""
        fi = open(filepath)
        model_data = cPickle.load(fi)
        fi.close()
        e1 = model_data['e1']
        e2 = model_data['e2']
        muscles = dict()
        for key in model_data.keys():
            if not(key in ['e1','e2']):
                muscles[key] = model_data[key]
        confocal_frame = Frame()
        confocal_frame['a1'] = e2[1]-e2[0]
        confocal_frame['a2'] = e1[1]-e2[0]
        confocal_frame['p'] = e2[0]
        self.construct(muscles,confocal_frame)

    def coords_from_frame(self,other_frame):
        ret = dict()
        for key in self.barycentric.keys():
            coords = np.dot(other_frame['A'],(self.barycentric[key]).T) + \
                     other_frame['p'][:,np.newaxis]
            ret[key] = coords
        return(ret)
    
        
class ModelView(object):
    def __init__(self,model):
        self.model = model
        self.plot_frame = copy.copy(model.frame)

    def plot(self,plot_frame,plotobject):
        lines = self.model.coords_from_frame(plot_frame)
        self.curves = list()
        for line in lines.values():
            self.curves.append(plotobject.plot(line[0,:],line[1,:]))

    def update_frame(self,plot_frame):
        lines = self.model.coords_from_frame(plot_frame)
        for curve,line in zip(self.curves,lines.values()):
            curve.setData(line[0,:],line[1,:])

    def frame_changed(self,roi):
        pnts = roi.saveState()['points']
        p = np.array(pnts[1])

        a1 = np.array(pnts[0])-p
        a2 = np.array(pnts[2])-p

        self.plot_frame['p'] = p
        self.plot_frame['a1'] = a1
        self.plot_frame['a2'] = a2
        self.update_frame(self.plot_frame)

