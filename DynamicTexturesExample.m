% This Script demonstrates the use of DMD on dynamic textures
% Written by Eric C. Hall - Sep 10 2014

load Water_params_forward
load Water_params_backward
n=length(Z0);   % Dimension of system state
p=length(C0);   % Dimension of observations
q=size(B,2);    % Dimension of driving noise
T=550;          % Number of video frames to be generated
D=5;            % std dev of observation noise
P_t=false(p,T);
theta=zeros(n,T);
full_data=zeros(p,T);
partial_data=zeros(p,T);

disp('Generating Data')
for t=1:T
    if t==1
        theta(:,t)=Z0;
    elseif (t>=100 && t<=120) || (t>=300 && t<=320)
        theta(:,t)=A_back*theta(:,t-1)+B_back*randn(q,1);
    else
        theta(:,t)=A*theta(:,t-1)+B*randn(q,1);
    end
    observe_vals=randperm(p)';
    P_t(:,t)=observe_vals>(.5*p);
    full_data(:,t)=C0+C*theta(:,t)+D*randn(p,1);
    partial_data(P_t(:,t),t)=full_data(P_t(:,t),t);
end
disp('done.')
disp('Performing DMD Algorithm')
[theta_hat_DMD loss_DMD]=DMD_Autoregressive(partial_data,A,C,C0,P_t,.5);
disp('Performing MD Algorithm')
[theta_hat_MD loss_MD]=DMD_Autoregressive(partial_data,eye(n),C,C0,P_t,.5);

% Plot Loss curves
y_min=min([loss_DMD loss_MD])*.95;
y_max=max([loss_DMD loss_MD])*1.05;
figure(1)
plot(1:T,loss_DMD,1:T,loss_MD,[100 100; 120 120; 300 300; 320 320],[y_min y_max],'k--')
axis([0 T y_min y_max])

%Video of true image, noisy image, missing data, DMD, MD and loss curve
vid_bool=lower(input('Create and store result video (about 1-1.5 GB) (y/n)? ','s'));
if strcmp(vid_bool,'y') || strcmp(vid_bool,'yes')
    disp('Creating video')
    h=figure(2);
    set(gcf,'position',[1 840 1020 840])
    aviobj=avifile('DynamicTextureResults.avi');
    for t=1:550;
        subplot(2,3,1), imagesc(reshape(C*theta(:,t)+C0,220,320),[0 255]); colormap bone
        axis off; title('True Image')
        subplot(2,3,2), imagesc(reshape(full_data(:,t),220,320),[0 255]);
        axis off; title('Noisy Observation')
        subplot(2,3,3), imagesc(reshape(partial_data(:,t),220,320),[0 255]);
        axis off; title('Noisy Observation, Missing data')
        subplot(2,3,4), imagesc(reshape(C*theta_hat_DMD(:,t)+C0,220,320),[0 255]);
        axis off; title('DMD Prediction')
        subplot(2,3,5), imagesc(reshape(C*theta_hat_MD(:,t)+C0,220,320),[0 255]);
        axis off; title('MD Prediction')
        subplot(2,3,6), plot(1:T,loss_DMD,1:T,loss_MD,...
            [100 100; 120 120; 300 300; 320 320],[y_min y_max],'k--',...
            t,loss_DMD(t),'ro',t,loss_MD(t),'ro')
        axis([0 T y_min y_max])
        title('Instantaneous Loss')
        legend('DMD','Mirror Descent',0)
        aviobj=addframe(aviobj,h);
        
    end
    aviobj=close(aviobj);
    disp('done.')
end

