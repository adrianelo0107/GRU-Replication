"""训练过程诊断：复用已有计算图，记录梯度冲突、表示和软关联，不改变优化目标。"""
import json
from pathlib import Path
import numpy as np
import torch
from .evaluation import json_ready


def module_group(name):
    return name.split('.')[0]


def gradient_comparison(named, mse_grads, cl_grads):
    result = {}
    for name, parameter in named:
        result.setdefault(module_group(name), [0., 0., 0.])
    for (name, _), a, b in zip(named, mse_grads, cl_grads):
        values = result[module_group(name)]
        if a is not None:
            values[0] += float(a.detach().square().sum())
        if b is not None:
            values[1] += float(b.detach().square().sum())
        if a is not None and b is not None:
            values[2] += float((a.detach()*b.detach()).sum())
    return {name:dict(mse_norm=float(np.sqrt(a)), weighted_cl_norm=float(np.sqrt(b)),
        weighted_cl_over_mse=float(np.sqrt(b/a)) if a > 0 else None,
        cosine=float(dot/np.sqrt(a*b)) if a > 0 and b > 0 else None)
        for name,(a,b,dot) in result.items()}


def summarize(model, history, batch, horizon_days):
    result = dict(stocks=len(batch.stock_ids), cl_pairs=int(batch.cl_mask.sum()))
    representations = {}
    for name in ['e_s','e_p','e_r','e_h','e_alpha']:
        if name in history:
            value=history[name].detach()
            representations[name]=dict(mean=float(value.mean()),std=float(value.std(unbiased=False)),
                mean_node_variance=float(value.var(dim=0,unbiased=False).mean()),
                mean_node_l2=float(value.norm(dim=1).mean()))
    result['representations']=representations
    if history.get('beta_h') is not None:
        beta=history['beta_h'].detach().cpu().numpy().astype(float)
        centered=beta-beta.mean(axis=0)
        eigen=np.maximum(np.linalg.eigvalsh(centered.T@centered/len(beta)),0)
        singular=np.sqrt(eigen)
        mass=singular/singular.sum() if singular.sum() > 0 else singular
        nonzero=mass[mass>0]
        edge_mass=beta.sum(axis=0)
        edge_mass=edge_mass/edge_mass.sum() if edge_mass.sum()>0 else edge_mass
        result['hidden_incidence']=dict(saturated_fraction=float(((beta<.01)|(beta>.99)).mean()),
            mean=float(beta.mean()),std=float(beta.std()),
            centered_effective_rank=float(np.exp(-(nonzero*np.log(nonzero)).sum())) if len(nonzero) else 0.,
            edge_mass_entropy=float(-(edge_mass*np.log(np.maximum(edge_mass,1e-30))).sum()),
            mean_node_degree=float(beta.sum(axis=1).mean()))
    predictions={'full':history['prediction'].detach()}
    with torch.no_grad():
        if hasattr(model,'heads'):
            for name,key in [('prior','e_p'),('hidden','e_h'),('alpha','e_alpha')]:
                if name in model.heads:
                    predictions[name]=model.heads[name](history[key]).detach()
    horizons={}
    for j,h in enumerate(horizon_days):
        mask=batch.label_mask[:,j]
        y=batch.target[mask,j].detach()
        entry={}
        for name,value in predictions.items():
            p=value[mask,j]
            if len(y)<2:
                continue
            cov=((p-p.mean())*(y-y.mean())).mean()
            denom=p.std(unbiased=False)*y.std(unbiased=False)
            entry[name]=dict(mean=float(p.mean()),variance=float(p.var(unbiased=False)),
                covariance=float(cov),ic=float(cov/denom) if float(denom)>1e-12 else None)
        if len(y):
            squared=(predictions['full'][mask,j]-y).square()
            top=max(1,int(np.ceil(len(y)*.01)))
            entry['target']=dict(max_abs_zscore=float(y.abs().max()),
                top_one_percent_error_fraction=float(squared.topk(top).values.sum()/squared.sum())
                    if float(squared.sum())>0 else 0.)
        horizons[str(h)]=entry
    result['horizons']=horizons
    result['batchnorm']={}
    for name,layer in model.named_modules():
        if isinstance(layer,torch.nn.BatchNorm1d) and layer.track_running_stats:
            result['batchnorm'][name]=dict(running_mean_rms=float(layer.running_mean.square().mean().sqrt()),
                running_var_min=float(layer.running_var.min()),running_var_mean=float(layer.running_var.mean()),
                running_var_max=float(layer.running_var.max()),batches=int(layer.num_batches_tracked))
    return result


class TrainingMonitor:
    def __init__(self, output, enabled, train_dates, valid_dates, count=3, horizons=(1,5,10,20)):
        self.enabled=enabled
        self.train_dates=set(self.select(train_dates,count))
        self.valid_dates=self.select(valid_dates,count)
        self.path=Path(output)/'training_monitor.jsonl'
        self.horizons=horizons
        if enabled:
            protocol=dict(train_dates=sorted(self.train_dates),valid_dates=self.valid_dates,
                horizon_days=list(horizons),sampling='fixed evenly spaced dates; not representative full-fold estimates',
                gradients='MSE versus gamma-weighted CL, original retained graph; before optimizer step',
                validation='history-only eval, no future encoder; original BN strategy',
                updates='no extra optimizer step; no new stochastic sampling; no test probes per epoch')
            (Path(output)/'monitor_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')

    @staticmethod
    def select(dates,count):
        if not dates:return []
        return [dates[i] for i in np.unique(np.linspace(0,len(dates)-1,min(count,len(dates)),dtype=int))]

    def trim_for_resume(self,epoch):
        if self.enabled and self.path.exists():
            rows=[line for line in self.path.read_text(encoding='utf-8').splitlines()
                  if json.loads(line)['epoch'] <= epoch]
            self.path.write_text('\n'.join(rows)+('\n' if rows else ''),encoding='utf-8')

    def training_record(self,model,history,batch,losses,epoch,weight):
        if not self.enabled or batch.date not in self.train_dates:return None
        named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        parameters=[p for _,p in named]
        a=torch.autograd.grad(losses['mse'],parameters,retain_graph=True,allow_unused=True)
        weighted=weight*losses['cl']
        b=torch.autograd.grad(weighted,parameters,retain_graph=True,allow_unused=True) if weighted.requires_grad else [None]*len(named)
        record=dict(epoch=epoch,split='train',date=batch.date,**summarize(model,history,batch,self.horizons))
        record['loss']={k:float(v.detach()) for k,v in losses.items()}
        record['gradients']=gradient_comparison(named,a,b)
        return record

    def append(self,record,model=None):
        if record is None:return
        if model is not None:
            norms={}
            for name,p in model.named_parameters():
                if p.grad is not None:
                    group=module_group(name)
                    norms[group]=norms.get(group,0.)+float(p.grad.detach().square().sum())
            record['actual_total_gradient_norm']={k:float(np.sqrt(v)) for k,v in norms.items()}
            parameter_norms={}
            for name,p in model.named_parameters():
                group=module_group(name)
                parameter_norms[group]=parameter_norms.get(group,0.)+float(p.detach().square().sum())
            record['parameter_norm']={k:float(np.sqrt(v)) for k,v in parameter_norms.items()}
        with self.path.open('a',encoding='utf-8') as file:
            file.write(json.dumps(json_ready(record),ensure_ascii=False,allow_nan=False)+'\n')

    @torch.no_grad()
    def validation(self,model,dataset,device,epoch):
        if not self.enabled:return
        if model.training:raise AssertionError('验证监测须在原 eval 模式，不改变 BN 策略')
        date_to_index={str(dataset.cache.dates[d]):i for i,d in enumerate(dataset.dates)}
        for date in self.valid_dates:
            batch=dataset[date_to_index[date]].to(device)
            history=model.forward_history(batch.x,batch.beta)
            self.append(dict(epoch=epoch,split='valid',date=date,**summarize(model,history,batch,self.horizons)))
