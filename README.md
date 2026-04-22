## The 8 clusters generated from data.py 
![Description](images/clusters.png)

## DDPM Forward Diffusion Process, final image generated from schedule.py, used the function from data.py to generate the testing sample.

![Description](images/ddpm_forward.png)

## Output of denoiser.py 
```bash
MLPDenoiser(input_dim=2, hidden_dim=256, time_emb_dim=32)
Parameters : 75,266

Input  x  : torch.Size([64, 2])
Output eps  : torch.Size([64, 2])  same shape as x
input_dim=   1  →  output torch.Size([64, 1])
input_dim=   4  →  output torch.Size([64, 4])
input_dim=  16  →  output torch.Size([64, 16])
input_dim= 128  →  output torch.Size([64, 128])
```

## Output of diffusion.py

```bash
MLPDenoiser(input_dim=2, hidden_dim=256, time_emb_dim=32)
Parameters: 75,266
Dataset size: 10,000   batch: 512   epochs: 500

 Epoch        Loss
────────────────────
     1    0.844528
    50    0.269171
   100    0.252601
   150    0.239369
   200    0.238406
   250    0.226165
   300    0.221924
   350    0.237555
   400    0.226080
   450    0.216588
   500    0.227314
────────────────────
Final loss : 0.227314
Loss curve  → /Users/ameygupta/sop_task/images/ddpm_loss.png
Checkpoint  → /Users/ameygupta/sop_task/images/ddpm_model.pt
```

## The Loss Curve generated from diffusion.py
![Description](images/ddpm_loss.png)

## Output of the 2nd part of diffusion.py (after implement unguided sampling)

```bash
Running reverse diffusion (2 000 samples) ...
Generated   : torch.Size([2000, 2])
Samples plot → /Users/ameygupta/sop_task/images/ddpm_samples.png
```

## The DDPM unguided sampling 
![Description](images/ddpm_samples.png)


## Output of verifiers.py

```bash
log_value      : torch.Size([16])  min=-2.353  max=-0.032
grad_log_value : torch.Size([16, 2])
closed-form == autograd 

GaussianMixtureVerifier: 
GaussianMixtureVerifier(K=8, std=0.1)
log_value : torch.Size([16])  min=-199.828  max=-1.637
grad_log_value : torch.Size([16, 2])
closed-form == autograd

log_value sanity: centres vs random :
At cluster centres : 0.000  (should be HIGH)
At random points   : -492.291  (should be LOWER)
log_value higher at modes

grad sanity: gradient points toward nearest centre: 
x         = [2.5, 0.0]
∇log p(x) = [-50.0, -0.0]  (should point toward [2, 0])
gradient direction correct
```