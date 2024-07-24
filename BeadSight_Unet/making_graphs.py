import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from BeadSight_Unet.model_training_old import MyDataset, plot_pressure_maps
from tqdm.auto import tqdm
import numpy as np
from Unet_model import UNet
import pickle


# def loss_plots(model_path):
#     model = torch.load(model_path)
#     training_loss = model['loss_values']  # Load training loss values
#     validation_losses = model['validation_losses']  # Load validation loss values
#     testing_losses = model['testing_losses']  # Load testing loss values
#     epoch_values = model['epoch_values']  # Load epoch values

#     for epoch in range(99):
#         print(f'{epoch}\t{training_loss[epoch]}\t{validation_losses[epoch]}\t{testing_losses[epoch]}')

#     plt.figure()
#     plt.plot(epoch_values, training_loss, label='Training Loss')
#     plt.plot(epoch_values, validation_losses, label='Validation Loss')
#     plt.title('Training & Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.savefig('temp.jpg')

def test(model_path):
    model = torch.load(model_path)
    print(model.keys())
    print(model['testing_losses'])

def inference_save(model_path, video_folder, pressure_folder):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the images to 256x256
        transforms.ToTensor(),  # Convert the images to PyTorch tensors
    ])

    # Check if CUDA is available and set the device to GPU, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    # Testing the model
    model = UNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create DataLoader
    # test_dataset = MyDataset(video_folder, pressure_folder, transform, mode='test')

    # try:
    #     # save the test_dataset to pickle to make it easier to load later
    #     with open('test_dataset.pkl', 'wb') as f:
    #         pickle.dump(test_dataset, f)
    #     print("I'm Pickle Rick!")
    # except:
    #     print('Error saving test_dataset,. Rick != Pickle')
    print('start load')
    import time
    start_time = time.time()
    with open('test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    print(f'Loaded in {time.time() - start_time} seconds')


    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1, prefetch_factor=6)

    criterion = torch.nn.MSELoss().to(device)


    # Initialize the loss
    total_test_loss = 0
    all_gen_pressure_maps = torch.empty((len(test_dataset), 1, 256, 256), dtype=torch.float32)
    all_gt_pressure_maps = torch.empty((len(test_dataset), 1, 256, 256), dtype=torch.float32)
    print(f'all_gen_pressure_maps shape: {all_gen_pressure_maps.shape}')
    all_test_file_names = []
    
    # generate the pressure maps from the ground_truth ones.
    idx_counter = 0
    for i, (video_frames, pressure_maps, test_file_name) in enumerate(tqdm(test_dataloader, desc=f'Testing')):
        video_frames, pressure_maps = video_frames.to(device), pressure_maps.to(device)

        print(f'video_frames shape: {video_frames.shape}')
        print(f'pressure_maps shape: {pressure_maps.shape}')
        # print(f'test_file_name: {test_file_name}')

        with torch.no_grad():
            # Forward pass
            predictions = model(video_frames)
            print(f'predictions shape: {predictions.shape}')

            all_gen_pressure_maps[idx_counter:idx_counter+predictions.size(0), :, :] = predictions.clone().detach().cpu()
            all_gt_pressure_maps[idx_counter:idx_counter+predictions.size(0), :, :] = pressure_maps.clone().detach().cpu()
            all_test_file_names.extend(test_file_name)
            idx_counter += predictions.size(0)

            # Calculate loss
            loss = criterion(predictions, pressure_maps)
            total_test_loss += loss.item() * video_frames.size(0)  # Multiply by batch size to accumulate correctly

    # Save the generated pressure maps
    torch.save(all_gen_pressure_maps, 'all_gen_pressure_maps.pth')
    torch.save(all_gt_pressure_maps, 'all_gt_pressure_maps.pth')
    np.save('all_test_file_names.npy', np.array(all_test_file_names))

def pressure_curve(pickle_path, finger_size_m = 0.0125):
    with open(pickle_path, 'rb') as f:
        run_data = pickle.load(f)
    
    forces = np.array(run_data['forces'])*(finger_size_m**2)*np.pi/4*10**3 # Force in N
    times = np.arange(len(forces))/30 # Time in seconds
    print(f'forces shape: {forces.shape}')
    print(f'forces: {forces}')

    mpl.rcParams['font.size'] = 16  # Adjust to change default font size for all text
    mpl.rcParams['axes.titlesize'] = 20  # Adjust to change the title font size specifically

    plt.figure()
    plt.plot(times, forces)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    # resize firgure
    plt.gcf().set_size_inches(4, 3.5)
    plt.savefig('Force_vs_Time.jpg', bbox_inches='tight', pad_inches = 1, dpi=300)
    print('saved')
    


def error_location_analysis():
    grid_size = 4
    # make 3x3 grid for analysis:
    total_errors = np.zeros((grid_size, grid_size))
    num_presses = np.zeros((grid_size, grid_size))

    # Load the generated pressure maps
    all_gen_pressure_maps = torch.load('data/all_gen_pressure_maps.pth').squeeze().detach().numpy()*1000 # kPa
    all_gt_pressure_maps = torch.load('data/all_gt_pressure_maps.pth').squeeze().detach().numpy()*1000
    all_test_file_names = np.load('data/all_test_file_names.npy')

    print(f'all_gen_pressure_maps shape: {all_gen_pressure_maps.shape}')
    print(f'all_gt_pressure_maps shape: {all_gt_pressure_maps.shape}')
    print(f'all_test_file_names shape: {len(all_test_file_names)}')

    for i in tqdm(range(all_gen_pressure_maps.shape[0])):
        if all_gt_pressure_maps[i].sum() == 0:
            continue

        for x in range(grid_size):
            for y in range(grid_size):
                if all_gt_pressure_maps[i][int(x*256/grid_size):int((x+1)*256/grid_size), int(y*256/grid_size):int((y+1)*256/grid_size)].sum() > 0:
                    num_presses[x, y] += 1
                    total_errors[x, y] += np.mean(np.abs(all_gen_pressure_maps[i] - all_gt_pressure_maps[i]))
                

    mpl.rcParams['font.size'] = 16  # Adjust to change default font size for all text
    plt.figure(figsize=(5, 5))
    plt.imshow(total_errors/num_presses, vmin=0)
    cbar = plt.colorbar(shrink=0.8)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Error (kPa)', rotation=270, labelpad=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('Error_vs_Location.jpg', dpi = 300)
    print('max error:', np.max(total_errors/num_presses))

    # plot all of the absolute errors:
    abs_errors = np.abs(all_gen_pressure_maps - all_gt_pressure_maps)
    mean_abs_errors = np.mean(abs_errors, axis=0)
    print("mae:", np.mean(abs_errors))

def mae(all_gen_pressure_maps, all_gt_pressure_maps):
    errors = all_gen_pressure_maps - all_gt_pressure_maps
    mae = np.mean(np.abs(errors))
    print(f'Mean Absolute Error: {mae}')

def mse(all_gen_pressure_maps, all_gt_pressure_maps):
    errors = all_gen_pressure_maps - all_gt_pressure_maps
    mse = np.mean(errors**2)
    print(f'Mean Squared Error: {mse}')
    print(f"Root Mean Squared Error: {np.sqrt(mse)}")


def total_force_error(all_gen_pressure_maps, all_gt_pressure_maps):
    predicted_total_force = np.sum(all_gen_pressure_maps, axis = (1, 2))
    gt_total_force = np.sum(all_gt_pressure_maps, axis = (1, 2))

    total_force_error = np.mean(np.abs(predicted_total_force - gt_total_force))/np.mean(gt_total_force)

    print('total force error:', total_force_error)

def OtzuIOU(all_gen_pressure_maps, all_gt_pressure_maps):
    # convert gen_pressure_maps to unit16 for opencv
    max_val = np.max(all_gen_pressure_maps, axis=(1, 2)).reshape(-1, 1, 1)
    normalized_gen = (all_gen_pressure_maps/max_val*65535).astype(np.uint16)

    import cv2
    gt_mask = np.zeros_like(all_gt_pressure_maps)
    gt_mask[np.where(all_gt_pressure_maps > 0)] = 1
    intersections = np.zeros(all_gen_pressure_maps.shape[0])
    unions = np.zeros(all_gen_pressure_maps.shape[0])
    for i in tqdm(range(all_gen_pressure_maps.shape[0])):
        # threshold with Otsu's method
        gen_mask = cv2.threshold(normalized_gen[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        intersections[i] = np.sum(np.minimum(gen_mask, gt_mask[i]))
        unions[i] = np.sum(np.maximum(gen_mask, gt_mask[i]))

    print('mask intersection over union (outside mean):', np.mean(intersections/unions))
    print('mask intersection over union (inside mean):', np.sum(intersections)/np.sum(unions))

def center_of_pressure_distance(all_gen_pressure_maps, all_gt_pressure_maps, sensor_size_mm = 38.7):
    x_locations = np.ones((256, 256)) * np.arange(256)/256
    y_locations = np.ones((256, 256)) * np.arange(256)[:, np.newaxis]/256 # unit normalized

    gt_center_x = (x_locations * all_gt_pressure_maps).sum(axis=(1, 2)) / all_gt_pressure_maps.sum(axis=(1, 2))
    gt_center_y = (y_locations * all_gt_pressure_maps).sum(axis=(1, 2)) / all_gt_pressure_maps.sum(axis=(1, 2))
    
    gen_center_x = (x_locations * all_gen_pressure_maps).sum(axis=(1, 2)) / all_gen_pressure_maps.sum(axis=(1, 2))
    gen_center_y = (y_locations * all_gen_pressure_maps).sum(axis=(1, 2)) / all_gen_pressure_maps.sum(axis=(1, 2))

    print('gt_x:', gt_center_x.shape)
    print('gt_y:', gt_center_y.shape)

    distances = np.sqrt((gt_center_x - gen_center_x)**2 + (gt_center_y - gen_center_y)**2)

    print('average distance:', np.mean(distances), np.mean(distances)*sensor_size_mm)


def inference_analysis(valid_cutoff=0):
    all_gen_pressure_maps = torch.load('data/all_gen_pressure_maps.pth').squeeze().detach().numpy()*1000 # kPa
    all_gt_pressure_maps = torch.load('data/all_gt_pressure_maps.pth').squeeze().detach().numpy()*1000
    all_test_file_names = np.load('data/all_test_file_names.npy')

    print('gen shape:', all_gen_pressure_maps.shape)
    print('gt shape:', all_gt_pressure_maps.shape)

    if valid_cutoff is not None:
        valid_indicies = np.where(all_gt_pressure_maps.max(axis=(1, 2)) > valid_cutoff)[0]
        all_gen_pressure_maps = all_gen_pressure_maps[valid_indicies]
        all_gt_pressure_maps = all_gt_pressure_maps[valid_indicies]
        print('gen shape:', all_gen_pressure_maps.shape)
        print('gt shape:', all_gt_pressure_maps.shape)

    mse(all_gen_pressure_maps, all_gt_pressure_maps)
    mae(all_gen_pressure_maps, all_gt_pressure_maps)
    total_force_error(all_gen_pressure_maps, all_gt_pressure_maps)
    if valid_cutoff is not None: # can't calculate total force error if there are no forces
        OtzuIOU(all_gen_pressure_maps, all_gt_pressure_maps)
        center_of_pressure_distance(all_gen_pressure_maps, all_gt_pressure_maps)




def plot_grasp_graphs(indexs, save_directory, rows=2, cols=2):
    # mpl.rcParams['font.size'] = 30  # Adjust to change default font size for all text
    # mpl.rcParams['axes.titlesize'] = 20  # Adjust to change the title font size specifically

    assert len(indexs) == rows*cols, 'Number of indexs must be equal to rows*cols'

    all_gen_pressure_maps = torch.load('data/all_gen_pressure_maps.pth')*1000
    all_gt_pressure_maps = torch.load('data/all_gt_pressure_maps.pth')*1000 # MPa to kPa

    fig = plt.figure(layout='constrained', figsize=(20, 6))
    plt.xticks([])
    plt.yticks([])

    subfigs_top = fig.subfigures(1, 2, width_ratios=[1, 1])
    cbar_ax = subfigs_top[1].subplots(1, 1)
    cbar_ax.axis('off')
    subfigs = subfigs_top[0].subfigures(rows, cols, wspace=0.05, hspace=0.05)
    print(f'subfigs shape: {subfigs.shape}')
    print(subfigs)

    cmin = np.min(all_gt_pressure_maps[indexs].numpy())
    cmax = np.max(all_gt_pressure_maps[indexs].numpy())

    all_axes = []
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            ground_truth = all_gt_pressure_maps[indexs[i]].squeeze().numpy()
            prediction = all_gen_pressure_maps[indexs[i]].squeeze().numpy()

            # Create a subplot with 1 row and 2 columns
            (ax1, ax2) = subfigs[r, c].subplots(1, 2)

            all_axes.append(ax1)
            all_axes.append(ax2)

            # Plot the ground truth pressure map
            img1 = ax1.imshow(ground_truth, cmap='gray', vmin=cmin, vmax=cmax)
            ax1.set_title('Ground Truth', fontsize=20)
            ax1.axis('off')  # Removes the axis for a cleaner look
            # ax1.xticks([])
            # ax1.yticks([])

            # Plot the predicted pressure map
            img2 = ax2.imshow(prediction, cmap='gray', vmin=cmin, vmax=cmax)
            ax2.set_title('Predicted', fontsize=20)
            ax2.axis('off')  # Removes the axis for a cleaner look
            # ax2.xticks([])
            # ax2.yticks([])

    # Add a color bar to the right of the subplots
    cbar = fig.colorbar(img1, ax=cbar_ax, orientation='vertical', location="left")
    cbar.set_label('Pressure (kPa)', rotation=270, labelpad=-70, fontsize=25)
    cbar.ax.tick_params(labelsize=20, labelleft=False, labelright=True, right=True, left=False)

    # # Save the plot
    # plt.tight_layout()

    # Save the figure
    plt.savefig(save_directory, bbox_inches='tight', dpi=300)

    # Close the figure
    plt.close(fig) 

def plot_vid_figs(frame_nums, path, save_path):

    for frame_num in tqdm(frame_nums):
        pressure_maps = np.load(path)
        plt.figure()
        plt.imshow(pressure_maps[frame_num], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        cbar = plt.colorbar()
        cbar.set_label('Pressure (kPa)', rotation=270)
        plt.gcf().set_size_inches(4, 3.5)
        plt.savefig(save_path + f'{frame_num}.jpg', bbox_inches='tight', pad_inches = 1, dpi=300)
        plt.close()

def calc_pressures_vid(indecies, path, sensor_width_m = 0.0387):
    pressure_maps = np.load(path)
    for i in indecies:
        print(f'{i}: {np.mean(pressure_maps[i])*1000*sensor_width_m**2}') #N
    
    pressure_maps = pressure_maps[min(indecies):max(indecies)]
    plt.figure()
    plt.plot((np.arange(0, pressure_maps.shape[0]) + min(indecies))/30, np.mean(pressure_maps, axis=(1, 2))*1000*sensor_width_m**2)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.savefig('force_vs_time.jpg')
    



if __name__ == '__main__':
    # print('All points:')
    inference_analysis(valid_cutoff=None)
    # print('\nValid points:')
    # inference_analysis(valid_cutoff=0)
    # print('\n20 cutoff')
    # inference_analysis(valid_cutoff=20)
    # print('Valid points with cutoff 1N:')
    # inference_analysis(valid_cutoff=8.15)
    # print('\nValid points with cutoff 2N:')
    # inference_analysis(valid_cutoff=16.3)  
    # exit()

    all_gen_pressure_maps = torch.load('data/all_gen_pressure_maps.pth').squeeze().detach().numpy()*1000 # kPa
    all_gt_pressure_maps = torch.load('data/all_gt_pressure_maps.pth').squeeze().detach().numpy()*1000

    print(np.max(all_gt_pressure_maps))

    model_path = 'best_model.pth'

    # test(model_path)
    # video_folder = 'data/processed_data/sensor_video_files'
    # pressure_folder = 'data/processed_data/sensor_pressure_files'
    # inference_save(model_path, video_folder, pressure_folder)

    plot_grasp_graphs([2000, 1000, 200, 300], 'ExampleResults.jpg', 2, 2)