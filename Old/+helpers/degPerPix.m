%{
Isolated version of dlg_calcvisangle.m from the EYE-EEG Toolbox, containing just the calculation
Computes the visual angle of one pixel in degrees
%}

function alpha_per_pix = degPerPix(screenwidth, resolution, viewingdist)
    %screenwidth : horizontal width of screen in mm
    %resolution  : horizontal resolution in pixels
    %viewingdist : viewing distance in mm

    mm_per_pix    = screenwidth / resolution;
    alpha_per_pix = (180/pi) * (2 * atan((mm_per_pix / 2) / viewingdist));
end