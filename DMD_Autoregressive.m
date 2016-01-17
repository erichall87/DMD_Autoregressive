function [theta_hat loss_DMD]=DMD_Autoregressive(data,A,C,C0,P_t,alpha,theta_min,theta_max,theta_hat1)
%DMD_Autoregressive - Runs DMD on data with given parameters.  
%
% See "Online Optimization in Dynamic Environments" (arXiv 1307.5944), 
% specifically section 7.1 by Eric C. Hall and Rebecca M. Willett
%
% This function uses the following dynamical model:
%
%   x_t = C0 + C*theta_t + w_t
%   theta_(t+1) = A*theta_t + B*u_t
%
% where x_t are the observations and theta is the true underlying state of 
% the system.  The function takes in the observations x_t and comes up with
% the streaming predictions of theta.
% 
% INPUT
%   data    - Observations of the system, each column is an observation
%   A       - State evolution matrix
%   C       - Sensing matrix
%
% OPTIONAL INPUT
%   C0          - Baseline observation intensity
%   P_t         - Indicator of observed values at each time
%   alpha       - Scalar to proportion step size parameter. eta_t=alpha/sqrt(t)
%   theta_min   - Minimum value any entry of theta prediction can take
%   theta_max   - Maximum value any entry of theta prediction can take
%   theta_hat1  - Prediction of theta at time t=1
%
% OUTPUT
%   theta_hat   - Predictions of theta, each column is a prediction
%   loss_DMD    - Time evolving instantaneous loss
%
% Code written: Sep 10 2014
% Written by: Eric C. Hall


%% Assign values to optional input
n=size(A,1);

if ~exist('theta_hat1','var')||isempty(theta_hat1)
    theta_hat1=zeros(n,1);
elseif ~all(size(theta_hat1)==[n 1])
    error('theta_hat1 must have as many rows as A')
end

if ~exist('theta_min','var')||isempty(theta_min)
    theta_min=-inf;
end

if ~exist('theta_max','var')||isempty(theta_max)
    theta_max=inf;
end

if ~exist('alpha','var')||isempty(alpha)
    alpha=1;
end

if ~exist('P_t','var')||isempty(P_t)
   P_t=ones(size(data));
end

if ~exist('C0','var')||isempty(C0)
   C0=zeros(size(data,1),1); 
end

%% Run DMD for given model

T=size(data,2);
loss_DMD=zeros(1,T);
theta_hat=zeros(n,T);
theta_hat(:,1)=theta_hat1;

for t=1:T
    if mod(t,50)==0
        disp(['t=' int2str(t) ' out of ' int2str(T)]); pause(.01);
    end
    eta=alpha/sqrt(t);
    %Incur loss
    DMD_image=C*theta_hat(:,t)+C0;
    loss_DMD(t)=norm(DMD_image(P_t(:,t))-data(P_t(:,t),t),2)^2;
    
    if t<T
        % Mirror Descent Step
        theta_tilde=(eye(n)-eta*C(P_t(:,t),:)'*C(P_t(:,t),:))*theta_hat(:,t) + eta* C(P_t(:,t),:)'*(data(P_t(:,t),t)-C0(P_t(:,t)));
        % Project to feasible set
        if ~isinf(theta_min)
            theta_tilde=theta_tilde.*(theta_tilde>=theta_min)+theta_min.*(theta_tilde<theta_min);
        end
        if ~isinf(theta_max)
            theta_tilde=theta_tilde.*(theta_tilde<=theta_max)+theta_max.*(theta_tilde>theta_max);
        end
        % Apply dynamics
        theta_hat(:,t+1)=A*theta_tilde;
    end
end
