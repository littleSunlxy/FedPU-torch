import torch
import torch.nn as nn
import torch.nn.functional as F


class MPULoss(nn.Module):
    def __init__(self, k, PiW, PkW, UiW, UkW):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.piW = PiW
        self.pkW = PkW
        self.uiW = UiW
        self.ukW = UkW

    def forward(self, outputs, labels, prior):
        outputs = outputs.cuda().float()
        outputs_Soft = F.softmax(outputs, dim=1)
        # 数据划分
        P_mask = (labels < self.numClass - 1).nonzero().view(-1)
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels >= self.numClass - 1).nonzero().view(-1)
        outputsU = torch.index_select(outputs, 0, U_mask)
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)

        # 计算目标
        ui = sum(-torch.log(1-outputsU_Soft[:, 0:self.numClass-1]+0.01)) / ((self.numClass - 1)*outputsU.size(0))
        uk = sum(-torch.log(outputsU_Soft[:, self.numClass-1]+0.01)) / outputsU.size(0)

        UnlabeledLossI = sum(ui)
        UnlabeledLossK = uk

        crossentropyloss=nn.CrossEntropyLoss()
        PositiveLossI = crossentropyloss(outputsP, labelsP)
        PositiveLossK = sum(-torch.log(1-outputsP_Soft[:, self.numClass-1]+0.01)) * prior

        objective = PositiveLossI*self.piW + PositiveLossK*self.pkW + UnlabeledLossI*self.uiW + UnlabeledLossK*self.ukW #w将三者统一到同一个数量级上
        # print("\n 1")
        # print(PositiveLossI*self.piW, PositiveLossK*self.pkW)
        # print(UnlabeledLossI*self.uiW, UnlabeledLossK*self.ukW)
        return objective, PositiveLossI*self.piW + PositiveLossK*self.pkW, UnlabeledLossI*self.uiW + UnlabeledLossK*self.ukW



class PLoss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()

    def forward(self, outputs, labels):
        outputs = outputs.cuda().float()

        # 数据划分
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1)
        labelsP = torch.index_select(labels, 0, P_mask).cuda()
        outputsP = torch.index_select(outputs, 0, P_mask).cuda()

        crossentropyloss=nn.CrossEntropyLoss().cuda()

        crossloss = crossentropyloss(outputsP, labelsP)
        return crossloss



class MPULoss_INDEX(nn.Module):
    def __init__(self, k, puW):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.puW = puW

    def forward(self, outputs, labels, priorlist, indexlist):
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)
        # 数据划分
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)
        # import pdb; pdb.set_trace()


        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        outputsU = torch.index_select(outputs, 0, U_mask)               #  unlabeldata 的 ground truth. setting限制不能使用
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)     #  所有在 unlabeldata 上的预测值

        PULoss = torch.zeros(1).cuda()

        for i in range(self.numClass):
            if i in indexlist:      # calculate ui
                pu3 = sum(-torch.log(1 - outputsU_Soft[:, i] + 0.01)) / \
                      max(1, outputsU.size(0)) / len(indexlist)
                PULoss += pu3
            else:
                pu1 = sum(-torch.log(1 - outputsP_Soft[:, i] + 0.01)) * \
                      priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass-len(indexlist))
                PULoss += pu1

        pu2 = torch.zeros(1).cuda()
        for index, i in enumerate(labelsP):   # need to be optimized
            x = outputsP_Soft[index][i]
            pu2 += -torch.log(1 - x + 0.01) * priorlist[i]

        PULoss -= pu2 / max(1, outputsP.size(0))

        crossentropyloss=nn.CrossEntropyLoss()
        crossloss = crossentropyloss(outputsP, labelsP)

        # objective = PULoss * self.puW
        objective = crossloss
        # objective = PULoss * self.puW + crossloss

        return objective, PULoss * self.puW, crossloss


class MPULoss_V2(nn.Module):
    def __init__(self, k, puW):
        super().__init__()
        self.numClass = torch.tensor(k).cuda()
        self.puW = puW

    def forward(self, outputs, labels, priorlist, indexlist):
        outputs = outputs.float()
        outputs_Soft = F.softmax(outputs, dim=1)
        new_P_indexlist =  torch.zeros(self.numClass).cuda()
        # import pdb; pdb.set_trace()
        indexlist = indexlist.long()
        for i in indexlist:
            new_P_indexlist[i] += 1
        # 数据划分
        P_mask = (labels <= self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        labelsP = torch.index_select(labels, 0, P_mask)
        outputsP = torch.index_select(outputs, 0, P_mask)
        outputsP_Soft = torch.index_select(outputs_Soft, 0, P_mask)

        U_mask = (labels > self.numClass - 1).nonzero(as_tuple=False).view(-1).cuda()
        outputsU = torch.index_select(outputs, 0, U_mask)               #  unlabeldata 的 ground truth. setting限制不能使用
        outputsU_Soft = torch.index_select(outputs_Soft, 0, U_mask)     #  所有在 unlabeldata 上的预测值

        PULoss = torch.zeros(1).cuda()


        # outputsU_Soft.size()/ outputsU.size() : [1011, 10]
        PU3 = torch.zeros(1).cuda()
        PU1 = torch.zeros(1).cuda()
        for i in range(self.numClass):
            if i in indexlist:      # calculate ui
                pu3 = sum(-torch.log(1 - outputsU_Soft[:, i] + 0.01)) / \
                      max(1, outputsU.size(0)) / len(indexlist)
                PU3 += pu3
            else:
                pu1 = sum(-torch.log(1 - outputsP_Soft[:, i] + 0.01)) * \
                      priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass-len(indexlist))
                PU1 += pu1

        import pdb; pdb.set_trace()
        pu3 = sum(-torch.log(1 - outputsU_Soft + 0.01) * new_P_indexlist) / \
                              max(1, outputsU.size(0)) / len(indexlist)
        pu1 = sum(-torch.log(1 - outputsP_Soft + 0.01) * new_P_indexlist) * \
             priorlist[indexlist[0]] / max(1, outputsP.size(0)) / (self.numClass-len(indexlist))
        print("puloss 3:", PU3, pu3)
        print("puloss 1:", PU1, pu1)


        pu2 = torch.zeros(1).cuda()
        for index, i in enumerate(labelsP):   # need to be optimized
            x = outputsP_Soft[index][i]
            pu2 += -torch.log(1 - x + 0.01) * priorlist[i]

        PULoss -= pu2 / max(1, outputsP.size(0))

        crossentropyloss=nn.CrossEntropyLoss()
        crossloss = crossentropyloss(outputsP, labelsP)

        objective = PULoss * self.puW
        # objective = PULoss * self.puW + crossloss

        return objective, PULoss * self.puW, crossloss
