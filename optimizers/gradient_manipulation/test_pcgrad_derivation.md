```
>>> inputs = torch.tensor([[ 1.54099607, -0.29342890], [ 0.66135216,  0.26692411]])
>>> l1w = torch.tensor([-0.00529398, +0.37932289, -0.58198076, -0.52038747]).reshape(2, 2)
>>> l1b = torch.tensor([-0.27234524, +0.18961589])
>>> act1 = inputs @ l1w.T + l1b
>>> act1.flatten()
tensor([-0.39180756, -0.55451745, -0.17459601, -0.33418226])
>>> act2 = torch.where(act1 >= 0, act1, 0)
>>> act2.flatten()
tensor([0., 0., 0., 0.])
>>> l2w = torch.tensor([-0.01401003, +0.56065750, -0.06275152, +0.18710934]).reshape(2, 2)
>>> l2b = torch.tensor([-0.21369691, -0.13899271])
>>> act3 = act2 @ l2w.T + l2b
>>> act3.flatten()
tensor([-0.21369691, -0.13899271, -0.21369691, -0.13899271])
>>> act4 = torch.where(act3 >= 0, act3, 0)
>>> act4.flatten()
tensor([0., 0., 0., 0.])
>>> l3w = torch.tensor([-0.67553341, -0.46830416, -0.29148576, +0.02619376]).reshape(2, 2)
>>> l3b = torch.tensor([+0.27954420, +0.42428017])
>>> act5 = act4 @ l3w.T + l3b
>>> act5.flatten()
tensor([0.27954420, 0.42428017, 0.27954420, 0.42428017])
>>> act6 = torch.where(act5 >= 0, act5, 0)
>>> act6.flatten()
tensor([0.27954420, 0.42428017, 0.27954420, 0.42428017])
>>> l4w = torch.tensor([-0.47937673, -0.30791873, +0.25683329, +0.58717293]).reshape(2, 2)
>>> l4b = torch.tensor([-0.14552270, +0.52913630])
>>> act7 = act6 @ l4w.T + l4b
>>> act7.flatten()
tensor([-0.41017354,  0.85005838, -0.41017354,  0.85005838])
>>> act8 = torch.where(act7 >= 0, act7, 0)
>>> act8.flatten()
tensor([0.00000000, 0.85005838, 0.00000000, 0.85005838])
>>> l5w = torch.tensor([-0.11397384, +0.07482187, +0.64026839, -0.65596199]).reshape(2, 2)
>>> l5b = torch.tensor([-0.44515052, -0.17901486])
>>> act9 = act4 @ l5w.T + l5b
>>> act9.flatten()
tensor([-0.44515052, -0.17901486, -0.44515052, -0.17901486])
>>> act10 = torch.where(act9 >= 0, act9, 0)
>>> act10.flatten()
tensor([0., 0., 0., 0.])
>>> l6w = torch.tensor([-0.27563018, +0.61094081, -0.45833221, -0.32550448]).reshape(2, 2)
>>> l6b = torch.tensor([-0.49401340, -0.66224861])
>>> act11 = act10 @ l6w.T + l6b
>>> act11.flatten()
tensor([-0.49401340, -0.66224861, -0.49401340, -0.66224861])
>>> act12 = torch.where(act11 >= 0, act11, 0)
>>> act12.flatten()
tensor([0., 0., 0., 0.])
```

```
>>> grad_act12 = act12 - labels['task2']
>>> grad_act12.flatten()
tensor([1.08452237, 1.39859545, 0.45190597, 0.16613023])
>>> grad_act11 = torch.where(act11 >= 0, 1, 0) * grad_act12
>>> grad_act11.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l6w = grad_act11.T @ act10
>>> grad_l6w.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l6b = grad_act11.sum(dim=0)
>>> grad_l6b
tensor([0., 0.])
```

```
>>> grad_act10 = grad_act11 @ l6w
>>> grad_act10.flatten()
tensor([0., 0., 0., 0.])
>>> grad_act9 = torch.where(act9 >= 0, 1, 0) * grad_act10
>>> grad_act9.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l5w = grad_act9.T @ act4
>>> grad_l5w.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l5b = grad_act9.sum(dim=0)
>>> grad_l5b
tensor([0., 0.])
```

```
>>> grad_act8 = act8 - labels['task1']
>>> grad_act8.flatten()
tensor([ 2.17878938,  0.28162712, -0.06167726,  0.22874105])
>>> grad_act7 = torch.where(act7 >= 0, 1, 0) * grad_act8
>>> grad_act7.flatten()
tensor([0.00000000, 0.28162712, -0.00000000, 0.22874105])
>>> grad_l4w = grad_act7.T @ act6
>>> grad_l4w.flatten()
tensor([0.00000000, 0.00000000, 0.14267047, 0.21653908])
>>> grad_l4b = grad_act7.sum(dim=0)
>>> grad_l4b
tensor([0.00000000, 0.51036817])
```

```
>>> grad_act6 = grad_act7 @ l4w
>>> grad_act6.flatten()
tensor([0.07233122, 0.16536382, 0.05874832, 0.13431056])
>>> grad_act5 = torch.where(act5 >= 0, 1, 0) * grad_act6
>>> grad_act5.flatten()
tensor([0.07233122, 0.16536382, 0.05874832, 0.13431056])
>>> grad_l3w = grad_act5.T @ act4
>>> grad_l3w.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l3b = grad_act5.sum(dim=0)
>>> grad_l3b
tensor([0.13107954, 0.29967439])
```

```
>>> grad_act4_task1 = grad_act5 @ l3w
>>> grad_act4_task1.flatten()
tensor([-0.09706335, -0.02954151, -0.07883607, -0.02399398])
>>> grad_act3_task1 = torch.where(act3 >= 0, 1, 0) * grad_act4_task1
>>> grad_act3_task1.flatten()
tensor([-0., -0., -0., -0.])
>>> grad_l2w_task1 = grad_act3_task1.T @ act2
>>> grad_l2w_task1.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l2b_task1 = grad_act3_task1.sum(dim=0)
>>> grad_l2b_task1
tensor([0., 0.])
```

```
>>> grad_act2_task1 = grad_act3_task1 @ l3w
>>> grad_act2_task1.flatten()
tensor([0., 0., 0., 0.])
>>> grad_act1_task1 = torch.where(act1 >= 0, 1, 0) * grad_act2_task1
>>> grad_act1_task1.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l1w_task1 = grad_act1_task1.T @ inputs
>>> grad_l1w_task1.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l1b_task1 = grad_act1_task1.sum(dim=0)
>>> grad_l1b_task1
tensor([0., 0.])
```

```
>>> grad_act4_task2 = grad_act9 @ l5w
>>> grad_act4_task2.flatten()
tensor([0., 0., 0., 0.])
>>> grad_act3_task2 = torch.where(act3 >= 0, 1, 0) * grad_act4_task2
>>> grad_act3_task2.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l2w_task2 = grad_act3_task2.T @ act2
>>> grad_l2w_task2.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l2b_task2 = grad_act3_task2.sum(dim=0)
>>> grad_l2b_task2.flatten()
tensor([0., 0.])
```

```
>>> grad_act2_task2 = grad_act3_task2 @ l3w
>>> grad_act2_task2.flatten()
tensor([0., 0., 0., 0.])
>>> grad_act1_task2 = torch.where(act1 >= 0, 1, 0) * grad_act2_task2
>>> grad_act1_task2.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l1w_task2 = grad_act1_task2.T @ inputs
>>> grad_l1w_task2.flatten()
tensor([0., 0., 0., 0.])
>>> grad_l1b_task2 = grad_act1_task2.sum(dim=0)
>>> grad_l1b_task2.flatten()
tensor([0., 0.])
```
