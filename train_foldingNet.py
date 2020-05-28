from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import numpy as np
# import models.PointNetFCAE as PointNetFCAE
from models.Folding import FlodingNet
import sys
sys.path.append('./models')
sys.path.append('./reference')
from kaolin.metrics.point import chamfer_distance
import utliz
from reference.dataset_shapenet_completion import PartDataset_Completion

parser = argparse.ArgumentParser('Completion Trainer')
parser.add_argument('--batch_size', type=int, help='batch size', default=64)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=9999)
parser.add_argument('--num_points', type=int, help='input batch size', default=2048)
parser.add_argument('--log_dir', type=str, help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--code_nfts', default=1024, type=int, help='Encoder output feature size')
args = parser.parse_args()

###################### ShapeNet Completion Dataset ########################
train_dataset = PartDataset_Completion(train=True, npoints=1024)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_dataset = PartDataset_Completion(train=False, npoints=1024)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
train_examples = len(train_dataset)
train_batches = len(train_dataloader)
test_examples = len(test_dataset)
test_batches = len(test_dataloader)
global plotter
plotter = utliz.VisdomLinePlotter(env_name='QH Completion_ShapeNet')
struc_name_list = ['ShapeNet']
strucNum = 0
###########################################################################

############################ Model & Optimizer ##############################
# Choose model #
completion_model = FlodingNet()
# completion_model = ConstrucNet()
# completion_model = ReshapeNet()
# completion_model = PCN()
# completion_model = Vanilla_GAN()
# completion_model = point_cloud_generator()
# discriminator_model = mlp_discriminator()
# discriminator_model.cuda()
# optimizer_D = optim.Adam(discriminator_model.parameters(), lr=1e-3)
completion_model.cuda()
optimizer = optim.Adam(completion_model.parameters(), lr=1e-4)
# optimizer = optim.Adagrad(completion_model.parameters(), lr=1e-2)
#############################################################################

grads = {}

epoch_base = 0
if args.log_dir != '':
    checkpoint = torch.load(args.log_dir)
    completion_model.load_state_dict(checkpoint['model_state_dict'])
    epoch_base = checkpoint['epoch']
print('training from epoch {}'.format(epoch_base))

print("Train examples: {}".format(train_examples))
print("Evaluation examples: {}".format(test_examples))
print("Start training...")
cudnn.benchmark = True

total_test_loss_min = 999
best_epoch = 0
for epoch in range(2002):
    print("--------Epoch {}--------".format(epoch+epoch_base))

    # train one epoch
    completion_model.train()
    total_train_loss_test = 0
    total_train_loss_repulsion = 0

    train_pc_loss = [utliz.AverageMeter(), utliz.AverageMeter()] #partial/pred CD loss

    for batch_idx, data in enumerate(train_dataloader, 0):
        point_partial_global, point_gt_global = data

        point_partial_global = point_partial_global.permute(0, 2, 1)  # Bs*3*1024
        point_partial_global, point_gt_global = Variable(point_partial_global.float(), requires_grad=True), Variable(point_gt_global)
        point_partial_global, point_gt_global = point_partial_global.cuda(), point_gt_global.cuda()

        optimizer.zero_grad()
        point_partial_global_center = point_partial_global.mean(2).unsqueeze(1).repeat(1,1024,1).permute(0,2,1)
        point_partial_local = point_partial_global - point_partial_global_center

        ################################ FoldingNet ##################################
        _, pred_local = completion_model(point_partial_local)
        # move pred to partial or align pred with partial
        pred_global = pred_local + point_partial_global_center.permute(0, 2, 1) - pred_local.mean(1).unsqueeze(1)
        # loss of FoldingNet
        for i in range(point_partial_global.size()[0]):
            if i==0:
                loss = chamfer_distance(pred_global[i], point_gt_global[i])
            else:
                loss += chamfer_distance(pred_global[i], point_gt_global[i])

        train_pc_loss[1].update(loss.item())
        loss.backward()
        optimizer.step()
        ##############################################################################

        if False:
            print(
            "batch: {}/{}, Train loss: {:.4f}".format(batch_idx, len(train_dataloader), loss.item() ))
    # torch.save({
    #     'epoch': epoch+epoch_base,
    #     'model_state_dict': completion_model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': loss,
    #             }, './checkpoints/epoch_'+str(epoch+epoch_base)+'.pt')
    print("Train pc loss: {:.4f}".format(train_pc_loss[1].avg))
    plotter.plot('Epoch_loss', 'train_cd',           'cd loss & dice', epoch, train_pc_loss[1].sum)

    if  (epoch % 20 == 0):
        color_grid = utliz.get_color()
        plotter.scatter('pointcloud_train_pred' + str(epoch), 'train', 'pointcloud_train_pred' + str(epoch), pred_global[0, :, :].data,
                        size=1, color=np.round(color_grid.reshape(-1, 3) * 255))#np.array([0]))
        plotter.scatter('pointcloud_train_part' + str(epoch), 'train', 'pointcloud_train_part' + str(epoch), point_partial_global.permute(0,2,1)[0, :, :].data,
                        size=1, color=np.array([[100, 0, 0]]))
        plotter.scatter('pointcloud_train_gt' + str(epoch), 'train', 'pointcloud_train_gt' + str(epoch), point_gt_global[0, :, :].data,
                        size=1, color=np.array([[0, 0, 100]]))


    ### EVAL ###
    completion_model.eval()
    test_pc_loss = [utliz.AverageMeter(), utliz.AverageMeter()] #partial/pred CD loss

    for batch_idx, data in enumerate(test_dataloader, 0):
        point_partial_global, point_gt_global = data

        point_partial_global = point_partial_global.permute(0, 2, 1)  # Bs*3*2048
        # point_gt = point_gt.permute(0, 2, 1)  # Bs*3*2048
        point_partial_global, point_gt_global = Variable(point_partial_global.float()), Variable(point_gt_global)
        point_partial_global, point_gt_global = point_partial_global.cuda(), point_gt_global.cuda()

        point_partial_global_center = point_partial_global.mean(2).unsqueeze(1).repeat(1,1024,1).permute(0,2,1)
        point_partial_local = point_partial_global - point_partial_global_center

        ################################ FoldingNet ##################################
        _, pred_local = completion_model(point_partial_local)
        # move pred to partial or align pred with partial
        pred_global = pred_local + point_partial_global_center.permute(0, 2, 1) - pred_local.mean(1).unsqueeze(1)
        # loss of FoldingNet
        for i in range(point_partial_global.size()[0]):
            if i==0:
                loss = chamfer_distance(pred_global[i], point_gt_global[i])
                # loss_partial = chamfer_distance(point_partial_global.permute(0,2,1)[i], point_gt_global[i])
            else:
                loss += chamfer_distance(pred_global[i], point_gt_global[i])
                # loss_partial += chamfer_distance(point_partial_global.permute(0,2,1)[i], point_gt_global[i])
        test_pc_loss[1].update(loss.item())
        # total_test_loss_partial += loss_partial.item()
        ##############################################################################


        # if batch_idx % 1000 == 1:
        if False:
            print(
                "batch: {}/{}, Train loss: {:.4f}".format(
                    batch_idx, len(train_dataloader), loss.item()))
    # torch.save({
    #     'epoch': epoch+epoch_base,
    #     'model_state_dict': completion_model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': loss,
    #             }, './checkpoints/epoch_'+str(epoch+epoch_base)+'.pt')

    print("MIN LOSS: ", total_test_loss_min, "BEST EPOCH: ", best_epoch)

    print("Test loss: {:.4f}".format(test_pc_loss[1].avg ))
    plotter.plot('Epoch_loss', 'test_cd',           'cd loss & dice', epoch, test_pc_loss[1].sum)

    if (epoch % 100 == 1):
        plotter.scatter('pointcloud_test_pred' + str(epoch), 'test', 'pointcloud_test_pred' + str(epoch),
                        pred_global[0, :, :].data,
                        size=1, color=np.round(color_grid.reshape(-1, 3) * 255))#np.array([0]))
        plotter.scatter('pointcloud_test_part' + str(epoch), 'train', 'pointcloud_test_part' + str(epoch),
                        point_partial_global.permute(0, 2, 1)[0, :, :].data,
                        size=1, color=np.array([[100, 0, 0]]))
        plotter.scatter('pointcloud_test_gt' + str(epoch), 'test', 'pointcloud_test_gt' + str(epoch),
                        point_gt_global[0, :, :].data,
                        size=1, color=np.array([[0, 0, 100]]))

print('END')

plotter_pause = utliz.VisdomLinePlotter(env_name='Completion_debug'+struc_name_list[strucNum])
plotter_pause.scatters2D = {}
plotter_pause.scatter2D('pc_xy', 'gt', 'gt', x=point_gt[0,:,0:2],color=[0,255,0],size=1)
plotter_pause.scatter2D('pc_xy', 'partial', 'partial', x=point_partial[0,0:2,:].permute(1,0),color=[255,0,0],size=2)
plotter_pause.scatter2D('pc_xy', 'pred', 'pred', x=pred[0,:,0:2].data,color=[0,0,255],size=2)

i_ = 2
plotter_pause.scatters2D = {}
plotter_pause.scatter2D('pc_xy', 'gt', 'gt', x=point_gt[i_,:,0:2],color=[0,255,0],size=1)
plotter_pause.scatter2D('pc_xy', 'partial', 'partial', x=point_partial[i_,0:2,:].permute(1,0),color=[255,0,0],size=2)
plotter_pause.scatters2D = {}
plotter_pause.scatter2D('pc_xy', 'gt', 'gt', x=point_gt[i_,:,0:2],color=[0,255,0],size=1)
plotter_pause.scatter2D('pc_xy', 'pred', 'pred', x=pred[i_,:,0:2].data,color=[0,0,255],size=2)

for i in range(8):
    plotter_pause.scatters2D = {}
    plotter_pause.scatter2D('pc_xy_gpar_'+str(i), 'gt', 'gpar_'+str(i), x=point_gt_global[i, :, 0:2], color=[0, 255, 0], size=1)
    plotter_pause.scatter2D('pc_xy_gpar_'+str(i), 'partial', 'gpar_'+str(i), x=point_partial_global[i, 0:2, :].permute(1, 0), color=[255, 0, 0],
                            size=2)
    plotter_pause.scatter2D('pc_xy_gpre_' + str(i), 'gt', 'gpre_'+str(i), x=point_gt_global[i, :, 0:2], color=[0, 255, 0], size=1)
    plotter_pause.scatter2D('pc_xy_gpre_'+str(i), 'pred', 'gpre_'+str(i), x=pred_global[i, :, 0:2].data, color=[0, 0, 255], size=2)

    plotter_pause.scatter2D('pc_xy_pp_' + str(i), 'partial', 'pp_'+str(i), x=point_partial_global[i, 0:2, :].permute(1, 0),
                            color=[255, 0, 0],
                            size=2)
    plotter_pause.scatter2D('pc_xy_pp_' + str(i), 'pred', 'pp_'+str(i), x=pred_global[i, :, 0:2].data, color=[0, 0, 255], size=2)


for i in range(8):
    plotter_pause.scatters = {}
    plotter_pause.scatter('pc_xy_gpar_' + str(i), 'gpar', 'gpar_' + str(i),
                          x=torch.cat((point_gt, point_partial.permute(0, 2, 1)), 1)[i, :, :].data,
                          color=np.array([[255, 0, 0], [0, 255, 0]]).repeat(1024, 0), size=2)
    plotter_pause.scatter('pc_xy_gpre_' + str(i), 'gpre', 'gpre_' + str(i),
                          x=torch.cat((point_gt, pred), 1)[i, :, :].data,
                          color=np.array([[255, 0, 0], [0, 0, 255]]).repeat(1024, 0), size=2)
    plotter_pause.scatter('pc_xy_pp_' + str(i), 'pp', 'pp_' + str(i),
                          x=torch.cat((point_partial.permute(0,2,1), pred), 1)[i, :, :].data,
                          color=np.array([[0, 255, 0], [0, 0, 255]]).repeat(1024, 0), size=2)

point_gt_partial_pred = torch.cat((point_gt, point_partial.permute(0, 2, 1), pred), 2)
np.save('./results/pc_gt_partial_pred/pc_gt_partial_pred', point_gt_partial_pred.cpu().detach().numpy())
