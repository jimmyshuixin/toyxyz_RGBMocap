"""Off-thread asset reads and coalesced, bounded animation windows."""
from collections import OrderedDict
from pathlib import Path
from threading import Condition
import numpy as np
from PySide6.QtCore import QThread,Signal
from .preview_data import FrameArray

class CharacterLoader(QThread):
    ready=Signal(str,object)
    changed=Signal()
    failed=Signal(str,str)
    evicted=Signal(str)
    preparing=Signal(str)
    def __init__(self,parent=None):
        super().__init__(parent);self.condition=Condition();self.pending=OrderedDict();self.data={};self.windows={};self.stop=False;self.epoch=0;self.errors=set();self.texture_sources={}
    def request(self,path,frame=0,texture_source=None):
        key=str(path)
        with self.condition:
            if key in self.errors:return
            if texture_source is not None:self.texture_sources[key]=str(texture_source)
            self.pending[key]=(frame,self.epoch);self.condition.notify()
    def frame(self,path,index):
        key=str(path)
        with self.condition:
            window=self.windows.get(key)
            if window and window[0]<=index<window[0]+len(window[1]):
                # A final window already contains every remaining frame.
                # Reloading it emits ready/changed and can create a repaint loop,
                # especially for imported characters with a one-frame rest pose.
                if window[0]+len(window[1])<window[2] and index>=window[0]+len(window[1])*3//4:
                    self.pending[key]=(index,self.epoch);self.condition.notify()
                return window[1][index-window[0]]
        self.request(path,index);return None
    def reset(self):
        with self.condition:
            self.epoch+=1;self.pending.clear();self.windows.clear();self.errors.clear();self.texture_sources.clear();self.condition.notify()
    def close(self):
        with self.condition:self.stop=True;self.condition.notify()
        self.wait()
        with self.condition:
            self.pending.clear();self.windows.clear();self.data.clear();self.texture_sources.clear()
    def run(self):
        arrays=OrderedDict();epoch=-1
        try:
            while True:
                with self.condition:
                    while not self.stop and not self.pending and epoch==self.epoch:self.condition.wait()
                    if self.stop:return
                    if epoch!=self.epoch:
                        for a in arrays.values():a.close()
                        arrays.clear();self.data.clear();epoch=self.epoch
                    if not self.pending:continue
                    key,(index,requested)=self.pending.popitem(last=False)
                try:
                    if key not in arrays:
                        while arrays and (len(arrays)>=8 or sum(sum(v.nbytes for v in asset.values() if hasattr(v,'nbytes')) for asset in self.data.values())>256*1024**2):
                            old,a=arrays.popitem(last=False);a.close();self.data.pop(old,None)
                            with self.condition:self.windows.pop(old,None)
                            self.evicted.emit(old)
                        p=Path(key)
                        if (p/'preview.npz').stat().st_size>128*1024**2:raise ValueError('Character preview exceeds the 128 MB asset limit.')
                        with np.load(p/'preview.npz') as archive:asset={k:archive[k] for k in archive.files}
                        if sum(a.nbytes for a in asset.values())>192*1024**2:raise ValueError('Character preview exceeds memory limits.')
                        asset['key']=key
                        from .name_overlay import character_samples
                        asset['name_samples']=character_samples(asset)
                        from .character_textures import load,validate_geometry
                        source=str(Path(self.texture_sources.get(key,key)).resolve())
                        textures=next((v['textures'] for v in self.data.values() if v.get('textures',{}).get('key')==source),None)
                        if textures is None:
                            self.preparing.emit('Loading character textures...')
                            textures=load(source,asset,lambda:self.stop or requested!=self.epoch)
                        else:validate_geometry(textures['metadata'],asset)
                        asset['textures']=textures
                        # Shared images count once across rest/animated takes. Keep
                        # texture residency bounded independently of clip length.
                        def resident_bytes():
                            unique={v['textures']['key']:v['textures']['bytes'] for v in self.data.values()}
                            unique[source]=textures['bytes']
                            return sum(unique.values())
                        if resident_bytes()>2048*1024**2:
                            raise ValueError('Character textures exceed the 2 GB loaded-preview limit. Open fewer characters in this project.')
                        if self.stop or requested!=self.epoch:continue
                        a=FrameArray(p/'preview-frames.npy');arrays[key]=a
                        if a.shape[-2:]!=(4,4) or a.shape[1]>4096:raise ValueError('Invalid character animation palette')
                        self.data[key]=asset
                    a=arrays[key];arrays.move_to_end(key);index=max(0,min(a.shape[0]-1,index))
                    # At most 4 MB per active take, regardless of clip duration.
                    count=max(1,min(128,4*1024**2//(a.shape[1]*64)));start=max(0,index-count//4)
                    block=np.asarray(a[start:min(a.shape[0],start+count)]).copy()
                    with self.condition:
                        if requested!=self.epoch:continue
                        self.windows[key]=(start,block,a.shape[0])
                    self.ready.emit(key,dict(asset=self.data[key],frames=a.shape[0],palette=a.shape[1]))
                    self.changed.emit()
                except Exception as exc:
                    if self.stop or requested!=self.epoch:continue
                    with self.condition:self.errors.add(key)
                    self.failed.emit(key,str(exc))
        finally:
            for a in arrays.values():a.close()
