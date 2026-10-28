"""Settings roundtrip and native job argument checks without remote execution."""
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
import unittest

from autobehaver.desktop_settings import (defaults, specifications, validate_config,
    featureimp_arguments, featureimp_options, update_posepred_template)
from autobehaver.desktop_agent import write_draft, load_draft, read_json

ROOT = Path(__file__).resolve().parents[1]


class DraftTests(unittest.TestCase):
    def test_reject_invalid_and_preserve_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'draft.json'
            config={'dataset':tmp,'start_at':'features','pipeline_name':'demo'}
            write_draft(config,'pipeline',path)
            original=path.read_bytes()
            with self.assertRaises(FileExistsError): write_draft(config,'pipeline',path)
            self.assertEqual(original,path.read_bytes())
            self.assertEqual(load_draft(path)['config']['features_ram'],16)
            for bad in ({'unknown':1},{'pp_batch':-1},{'pp_use_tta':'false'},
                        {'pp_vram_fraction':.95},{'pp_vram_fraction':float('nan')},
                        {'features_cpu':False},{'model16':'/model.pth'},
                        {'pp_num_workers':99},{'feature_workers':99},
                        {'pp_tta_noise_level':.123456789012}, {'pipeline_name':'bad\nname'}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    validate_config({**config,**bad},'pipeline')
            path.write_text('{"dataset":"one","dataset":"two"}')
            with self.assertRaises(ValueError): read_json(path)

    def test_native_featureimp_flags_and_pose_template(self):
        from autobehaver.postinf.featureimp.config import build_arg_parser
        cfg={'fi_opt_MIN_VARIANCE_THRESHOLD':.0002,'fi_opt_n_splits':3,
             'fi_opt_fsea_fdr_threshold':.15,'fi_opt_fsea_permutations':1000}
        args=build_arg_parser().parse_args(featureimp_arguments(cfg))
        for key,value in featureimp_options(cfg).items(): self.assertEqual(getattr(args,key),value)
        content=(ROOT/'scripts/hpc/posepred.sbatch').read_text()
        changed=update_posepred_template(content,{'pp_batch':128,'pp_use_tta':False,'pp_vram_fraction':.7,'pp_min_pose_bout':7})
        for flag in ('--batch 128','--no_use_tta','--vram_fraction 0.7','--min_pose_bout 7'):
            self.assertIn(flag,changed)
        for key in ('batch','num_workers','use_tta','vram_fraction','noise_workers'):
            self.assertEqual(len(re.findall(r'--(?:no[-_])?'+key+r'\b',changed)),1)
        self.assertNotIn('--use_tta',changed)
        from autobehaver.preinf.posepred_inference.cli import build_arg_parser as pose_parser
        cli = shlex.split(changed.replace('\\\n',' ').split('autobehaver.preinf.posepred_inference ',1)[1])
        cli = [arg for arg in cli if arg != '${selection_args[@]}']
        parsed = pose_parser().parse_args(cli)
        self.assertEqual(parsed.batch,128)
        self.assertFalse(parsed.use_tta)
        self.assertEqual(parsed.vram_fraction,.7)

    def test_inputs_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp)/'Project with spaces'; project.mkdir()
            folder=project/'8_master_matrix'; folder.mkdir()
            matrix=folder/'MASTER_FEATURE_MATRIX.csv'
            matrix.write_text('individual,group,sex,value\na,WT,male,1\nb,Mut,male,2\nc,WT,female,3\n')
            cfg={'dataset':str(project),'group_a':'WT','group_b':'Mut','control_group':'WT',
                 'selected_identities':[['a','WT','male'],['b','Mut','male']]}
            valid=validate_config(cfg,'featureimp',check_inputs=True)
            self.assertEqual(valid['matrix'],str(matrix))
            for bad in ({'selected_identities':[['a','WT','male']]}, {'sex':'female'},
                        {'selected_identities':[['missing','WT','male']]},
                        {'fi_opt_fsea_min_size':501,'fi_opt_fsea_max_size':500}):
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    validate_config({**cfg,**bad},'featureimp',check_inputs=True)

    def test_native_builds_receive_advanced_options(self):
        from autobehaver import desktop_jobs as jobs
        from autobehaver.postinf.featureimp.config import build_arg_parser
        from autobehaver.resource_monitor import add_monitor_arguments
        from autobehaver.cli import build_parser
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); project=root/'data/project with spaces'; project.mkdir(parents=True)
            (root/'bundles').mkdir()
            matrix=project/'matrix.csv'
            matrix.write_text('individual,group,sex,Kinematic_speed\na,WT,MALE,1\nb,Mut,MALE,2\nc,Extra,FEMALE,3\n')
            cfg=dict(kind='featureimp',run_name='check',dataset=str(project),data_local=str(root/'data'),
                data_remote='/data/test',jobs_local=str(root/'bundles'),jobs_remote='/data/bundles',
                python='/env/bin/python',resources='/resources',matrix=str(matrix),group_a='WT',group_b='Mut',
                control_group='WT',sex='all',pca=True,fsea=True,fi_sharded=False,
                selected_identities=[['a','WT','MALE'],['b','Mut','MALE']],
                fi_opt_n_splits=3,fi_opt_fsea_fdr_threshold=.15,fi_opt_MIN_VARIANCE_THRESHOLD=.00002)
            record=jobs.build_job(cfg,ROOT,root/'local')
            target=Path(record['prepared']); plan=json.loads((target/'plan.json').read_text())
            args=add_monitor_arguments(build_arg_parser()).parse_args(plan['command'][3:])
            self.assertEqual(args.n_splits,3); self.assertEqual(args.fsea_fdr_threshold,.15)
            self.assertEqual(args.MIN_VARIANCE_THRESHOLD,.00002)
            self.assertEqual(len(jobs.matrix_inventory(target/'featureimp_matrix.csv')),2)
            self.assertFalse(Path(record['mounted']).exists())
            # Existing-DLC inventory: preparation only, no inference or scientific processing.
            dlc=project/'3_dlc'; dlc.mkdir()
            (dlc/'WT_male_exp_trial1_rpi-1_movie0_20261018120000PM_ch1DLC_model.csv').write_text('scorer,DLC,DLC,DLC\nbodyparts,head,head,head\ncoords,x,y,likelihood\n0,1,2,.9\n')
            pcfg=dict(cfg,kind='pipeline',start_at='features',speed='fast',moviechecker_mode='reviewed',
                      models=[],pp_batch=64,pp_vram_fraction=.65,pp_use_tta=False,
                      allocations={'features':[10,16],'posepred':[12,24]},feature_workers=8)
            pipeline=jobs.build_job(pcfg,ROOT,root/'local')
            template=(Path(pipeline['prepared'])/'templates/posepred.sbatch').read_text()
            for expected in ('--batch 64','--vram_fraction 0.65','--no_use_tta','--memory_gb 24'):
                self.assertIn(expected,template)
            metadata=project/'metadata.csv'
            for name in ('a.h5','a.labels.h5','b.h5','b.labels.h5'): (project/name).write_text('fixture')
            metadata.write_text('filename,labels,set\na.h5,a.labels.h5,train\nb.h5,b.labels.h5,val\n')
            training=jobs.build_job(dict(cfg,kind='training',training_metadata=str(metadata),
                training_mode='supervised',training_options={'batch':'128','num_workers':'2'},
                train_cpus=4,train_memory=8),ROOT,root/'local')
            training_plan=json.loads((Path(training['prepared'])/'plan.json').read_text())
            arguments=training_plan['command'][3:]
            parsed=build_parser().parse_args(arguments)
            self.assertEqual(parsed.batch,128); self.assertEqual(parsed.num_workers,2)

    @unittest.skipUnless(os.environ.get('AUTOBEHAVER_TEST_SCIENCE')=='1','Requires FeatureImp scientific dependencies.')
    def test_real_sharded_plan_freezes_settings(self):
        import csv
        from autobehaver import desktop_jobs as jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); project=root/'data/experiment'; project.mkdir(parents=True)
            (root/'bundles').mkdir()
            matrix=project/'matrix.csv'
            with matrix.open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['individual','group','sex']+[f'Kinematic_feature{i}' for i in range(4)])
                for i in range(24): writer.writerow([f'fly{i}','A' if i<12 else 'B','MALE']+[(i*(j+1)%17)+j/10 for j in range(4)])
            cfg=dict(kind='featureimp',run_name='fixture',dataset=str(project),data_local=str(root/'data'),
                data_remote='/data/test',jobs_local=str(root/'bundles'),jobs_remote='/data/bundles',
                python='/env/bin/python',resources='/resources',matrix=str(matrix),group_a='A',group_b='B',
                control_group='A',sex='all',pca=True,fsea=True,fi_sharded=True,fi_shards=2,
                fi_opt_n_splits=3,fi_opt_H_hits_per_feature=2,fi_opt_n_features_per_model=2,
                fi_opt_fsea_fdr_threshold=.15,fi_opt_fsea_permutations=1234,fi_opt_pca_fixed_k=3)
            try: record=jobs.build_job(cfg,ROOT,root/'local')
            except Exception:
                for log in (root/'local').rglob('local_planning.log'): print(log.read_text())
                raise
            target=Path(record['prepared'])
            saved=json.loads((target/'analysis_seed/FEATUREIMP_RUN_SETTINGS.json').read_text())
            for key,value in featureimp_options(cfg).items(): self.assertEqual(saved[key],value,key)
            manifest=json.loads((target/'analysis_seed/_featureimp_plan/PLAN_MANIFEST.json').read_text())
            self.assertEqual(manifest['training_row_count'],24)
            self.assertEqual(manifest['active_num_shards'],2)
            self.assertFalse(Path(record['mounted']).exists())


@unittest.skipUnless(os.environ.get('AUTOBEHAVER_TEST_QT')=='1','Set AUTOBEHAVER_TEST_QT=1 for offscreen GUI tests.')
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['QT_QPA_PLATFORM']='offscreen'
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_draft_roundtrip_all_workflows(self):
        from autobehaver.desktop import make_window
        from autobehaver.desktop_drafts import write_draft as gui_save
        from PySide6.QtWidgets import QToolButton, QSpinBox
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            prior=tmp/'prior.json'; prior.write_text('{"sentinel":true}')
            for kind,overrides in [
                ('pipeline',{'dataset':str(tmp/'space project'),'pp_batch':128,'pp_use_tta':False,'pp_tta_noise_level':.005,'pp_vram_fraction':.7}),
                ('featureimp',{'dataset':str(tmp),'group_a':'WT','group_b':'Mut','control_group':'WT','fi_override':True,
                              'fi_dataset':str(tmp),'matrix':str(tmp/'custom.csv'),'fi_opt_fsea_fdr_threshold':.1,
                              'selected_identities':[['a','WT','male'],['b','Mut','male']]}),
                ('training',{'training_directory':str(tmp),'train_opt_batch':'256','train_opt_base_lr':'0.00002'})]:
                with self.subTest(kind=kind):
                    path=write_draft(overrides,kind,tmp/(kind+'.json'))
                    state=tmp/(kind+' state'); state.mkdir()
                    (state/'last_run.json').write_text(json.dumps({'path':str(prior)}))
                    before=(state/'last_run.json').read_bytes()
                    # Opening must never prepare/send jobs or connect.
                    with patch('autobehaver.desktop.jobs.build_job',side_effect=AssertionError('built on open')):
                        window=make_window(state,initial_draft=(load_draft(path),path))
                    try:
                        self.assertIsNone(window.record)
                        self.assertFalse(window.session.connected)
                        self.assertFalse(window.submit_button.isEnabled())
                        self.assertEqual(before,(state/'last_run.json').read_bytes())
                        self.assertEqual(set(window.fields),set(specifications()))
                        decimal=window.fields['pp_vram_fraction']
                        self.assertEqual(decimal.textFromValue(.8),'0.8')
                        from PySide6 import QtCore, QtGui
                        before_value=decimal.value()
                        event=QtGui.QWheelEvent(QtCore.QPointF(2,2),QtCore.QPointF(2,2),QtCore.QPoint(),QtCore.QPoint(0,120),
                            QtCore.Qt.NoButton,QtCore.Qt.NoModifier,QtCore.Qt.NoScrollPhase,False)
                        self.app.sendEvent(decimal,event)
                        self.assertEqual(decimal.value(),before_value)
                        actual=window.values()
                        expected=load_draft(path)['config']
                        for key,value in actual.items():
                            if key in expected and key not in ('group_a','group_b','control_group'):
                                self.assertEqual(value,expected[key],key)
                        self.assertTrue(any(b.isChecked() and 'changed' in b.text() for b in window.findChildren(QToolButton)))
                        # Exercise editable spins rather than merely inspecting text.
                        spin=window.fields['features_cpu']; spin.setValue(11); spin.stepUp()
                        self.assertEqual(spin.value(),12); spin.setValue(expected['features_cpu'])
                        if kind=='featureimp':
                            self.assertEqual(window.saved_groups['group_a'],'WT')
                            self.assertEqual(window.pending_restored_selection['identities'],set(map(tuple,overrides['selected_identities'])))
                        window.save_run()
                        saved=read_json(window.draft_path)
                        if kind=='featureimp':
                            self.assertEqual(saved['config']['group_a'],'WT')
                            self.assertEqual(saved['config']['selected_identities'],overrides['selected_identities'])
                        for key in ('pp_batch','pp_use_tta','pp_tta_noise_level','fi_opt_fsea_fdr_threshold','train_opt_batch'):
                            self.assertEqual(saved['config'][key],expected[key])
                        self.assertEqual(prior.read_text(),'{"sentinel":true}')
                    finally:
                        for timer in (window.monitor_timer,window.fi_refresh_timer,window.fi_poll_timer,window.connection_timer): timer.stop()
                        window.deleteLater(); self.app.processEvents()


if __name__=='__main__': unittest.main()
