import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from skimage import measure

def main(image):
    clicked_points = []

    # callback function for clicking on image
    def on_click(event):
        # check that the zoom tool is not selected
        if fig.canvas.toolbar.mode == "":
            if event.button == 1:  # check for LMB click
                if len(clicked_points) < 2:
                    clicked_points.append((event.ydata, event.xdata))
                ax1.plot([point[1] for point in clicked_points], [point[0] for point in clicked_points], 'r*')
                ax1.plot([point[1] for point in clicked_points], [point[0] for point in clicked_points], 'r-')
                
                if len(clicked_points) == 2:
                    ax2.clear()
                    ax2.plot(measure.profile_line(image, clicked_points[0], clicked_points[1]))
                
                plt.draw()


    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image, cmap="gray")

    # Connect the callback function to the 'button_press_event' event
    ax1.figure.canvas.mpl_connect('button_press_event', on_click)

    # reset_selection = Button(plt.axes([0.7, 0.01, 0.2, 0.05]), "reset selection")
    # reset_selection.on_clicked(reset_selection_click)

    plt.show()

if __name__ == "__main__":
    image = np.load("/home/clarkcs/my_ctdata/R003065-QRM_micro-bar_step-mode_1257-projections_01/R003065-QRM_micro-bar_step-mode-1257-projections_vertical_target.npy")
    main(image)