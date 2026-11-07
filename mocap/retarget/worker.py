"""Transactional import/retarget jobs. A failed job never replaces a usable character."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
from mocap.storage import atomic_json
from .runtime import blender

def report(text):print('CHARACTER: '+text,flush=True)

def invoke(mode,out,*args):
    command=[str(blender()),'-b','--factory-startup','--disable-autoexec','--threads','3','--python-exit-code','1','--python',str(Path(__file__).with_name('blender_entry.py')),'--',mode,str(out),*map(str,args)]
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    with (out/'blender.log').open('a',encoding='utf8') as log:
        p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf8',errors='replace',env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        tail=[]
        for line in p.stdout:
            log.write(line);tail=(tail+[line])[-25:]
            if line.startswith('CHARACTER:'):print(line.strip(),flush=True)
        p.stdout.close()
        if p.wait():raise RuntimeError('Character processing failed:\n'+''.join(tail)[-2500:])

def main(args):
    # Packaged workers run with -I, which ignores PYTHONIOENCODING/PYTHONUTF8.
    # QProcess decodes these pipes as UTF-8, regardless of the Windows locale.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    request=Path(args[args.index('--request')+1]);job=json.loads(request.read_text(encoding='utf8'))
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    if job['mode']=='adopt':
        rows=[]
        for c in job['characters']:
            report('Saving imported character '+c['name'])
            shutil.copytree(Path(job['source'])/c['asset'],out/c['id']/'asset')
            saved=dict(c,asset=job['prefix']+'/'+c['id']+'/asset')
            if c.get('animation'):
                shutil.copytree(Path(job['source'])/c['animation'],out/c['id']/'animation')
                saved['animation']=job['prefix']+'/'+c['id']+'/animation'
            if c.get('face_motion'):
                face_out=out/c['id']/'face.npz'
                shutil.copy2(Path(job['source'])/c['face_motion'],face_out)
                saved['face_motion']=job['prefix']+'/'+c['id']+'/face.npz'
            rows.append(saved)
        result=dict(characters=rows)
    elif job['mode']=='import':
        invoke('import',out,job['file'])
        result=json.loads((out/'import.json').read_text(encoding='utf8'))
    elif job['mode']=='export':
        for i,(take,target) in enumerate(zip(job['takes'],job['files'])):
            report(f'Exporting character take {i+1}/{len(job["takes"])}')
            folder=Path(job['animation'])/take['folder']
            options=dict(format=job.get('format','fbx'),include_mesh=job.get('include_mesh',True))
            invoke('export',out,folder,target,json.dumps(take.get('transform')),job['source_name'],json.dumps(options))
        result=dict(files=job['files'])
    elif job['mode']=='face':
        from .face_track import load_track
        from .source import stamp
        face=Path(job['face_motion']);track=load_track(face)
        asset=Path(job['asset']);takes=[];channels=set()
        source_takes=job.get('takes') or [dict(start=track['start_seconds'],frames=len(track['weights']),fps=track['fps'])]
        for i,take in enumerate(source_takes):
            folder=out/f'take-{i}'
            existing=bool(job.get('takes'))
            if existing:
                base=Path(job['source_animation']).resolve()
                source=(base/take['folder']).resolve()
                if not source.is_relative_to(base):raise ValueError('Invalid character take path')
            else:source=asset
            inputs=[face,source/'import.json',source/('animation.blend' if existing else json.loads((source/'import.json').read_text(encoding='utf8'))['kind']+'-rest.blend')]
            before=stamp(inputs)
            report(f'Take {i+1}/{len(source_takes)}: copying character for facial animation')
            shutil.copytree(source,folder)
            options=dict(start=float(take['start']),frames=int(take['frames']),fps=float(take['fps']),rest=not existing)
            invoke('face',folder,face,json.dumps(options))
            if stamp(inputs)!=before:raise RuntimeError('Face capture or character changed while applying facial animation. Retry after processing finishes.')
            info=json.loads((folder/'bake.json').read_text(encoding='utf8'));channels.update(info.get('face_channels',[]))
            atomic_json(folder/'face-receipt.json',dict(source_stats=before,**options))
            takes.append(dict(take,folder=f'take-{i}'))
        result=dict(takes=takes,source_name=job.get('source_name','Face capture'),result=job.get('result','initial'),face_motion=str(face),face_channels=sorted(channels))
    else:
        from .source import prepare,stamp
        from .pipeline import run
        asset=Path(job['asset']);kind=json.loads((asset/'import.json').read_text(encoding='utf-8'))['kind'];takes=[]
        if kind=='pmx':
            from .pmx_validation import validate_asset
            validate_asset(asset)
        for i,take in enumerate(job['takes']):
            folder=out/f'take-{i}';folder.mkdir()
            report(f'Take {i+1}/{len(job["takes"])}: preparing selected source')
            for name in [f'{kind}-rest.blend',f'{kind}-rig.json',f'{kind}-canonical.npz',f'{kind}-contact-geometry.npz','import.json']:
                shutil.copy2(asset/name,folder/name)
            receipt=prepare(Path(take['project']),job['result'],folder)
            report(f'Take {i+1}/{len(job["takes"])}: solving contact retargeting')
            diagnostic=run(folder,kind)
            report(f'Take {i+1}/{len(job["takes"])}: baking body bones')
            invoke('bake',folder,kind)
            from mocap.face.pipeline import load_current
            face=Path(job['face_motion']) if job.get('face_motion') else load_current(take['project'])
            if face is not None:
                face_stamp=stamp([face])
                report(f'Take {i+1}/{len(job["takes"])}: applying captured facial expressions')
                invoke('face',folder,face,json.dumps(dict(start=float(take['start']),frames=receipt['frames'],fps=receipt['fps'],optional=not bool(job.get('face_motion')))))
                if stamp([face])!=face_stamp:raise RuntimeError('Face capture changed during retargeting.')
                receipt['source_stats'].update(face_stamp)
                skipped=folder/'face-skipped.json'
                receipt['face_capture']=json.loads(skipped.read_text(encoding='utf8')) if skipped.is_file() else dict(status='ready',source=str(face))
            if stamp(map(Path,receipt['source_stats']))!=receipt['source_stats']:
                raise RuntimeError('Source changed during retargeting. Apply your changes and retarget again.')
            atomic_json(folder/'receipt.json',dict(receipt,diagnostic=diagnostic))
            takes.append(dict(folder=f'take-{i}',start=take['start'],frames=receipt['frames'],fps=receipt['fps'],transform=take.get('transform'),source=take['project']))
        result=dict(takes=takes,source_name=job['source_name'],result=job['result'])
    result['seconds']=time.perf_counter()-start
    atomic_json(out/'complete.json',result)
    report('Complete')

if __name__=='__main__':main(sys.argv[1:])
