---
name: Training Help
about: Get help with training issues (policy collapse, slow progress, etc.)
title: "[TRAINING]"
labels: help wanted
assignees: ''

---

**Describe the issue**
What's happening with your training? (e.g. Mario stuck, policy collapse, no progress, etc.)

**Training details**
- **Level(s)**: [e.g. 1-1, or multi-level 1-1,1-2]
- **Current timesteps**: [e.g. 5M]
- **Training command used**:
python train_ppo.py ...

**Hardware**
- GPU: e.g. RTX 5090
- RAM: e.g. 64GB
- n_envs: e.g. 16
Model version
- [ ] v1.x (SIMPLE_MOVEMENT, 7 actions)
- [ ] v2.x (COMPLEX_MOVEMENT, 12 actions)

**What have you tried?**
- [ ] Ran diagnose_policy.py
- [ ] Checked TensorBoard logs
- [ ] Tried resuming from earlier checkpoint
- [ ] Adjusted entropy coefficient (--ent-coef)
- [ ] Other: describe

**Diagnosis output**
If you ran diagnose_policy.py, paste the output here:
paste output here

**TensorBoard screenshots**
If applicable, add screenshots of your training curves (reward, entropy, loss, etc.)

**Additional context**
Add any other context about the training issue here.
