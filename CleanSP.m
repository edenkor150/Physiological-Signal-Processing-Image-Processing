function out_I = CleanSP(in_I, Type, var1, var2)
    if strcmp(Type, 'Gaussian')
        h = fspecial('gaussian', var1, var2);
        out_I = filter2(h, in_I, 'same');
    elseif strcmp(Type, 'Median')
        out_I = medfilt2(in_I, [var1 var2]);
    else
        error('Invalid filter type');
    end
end