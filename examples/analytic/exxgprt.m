% EXXGPRT  Exact reflected field of a thin planar layer, for the infinite planar case.
%
% Computes the analytic (Sommerfeld-integral) reference trace of the infinite planar case: the
% electric field reflected by an unbounded planar fluid-filled layer in a homogeneous host,
% recorded by a receiver dipole parallel to the transmitter dipole and to the layer, 1.0 m from
% it along the dipole axis.
%
% Geometry (the layer normal is the vertical axis here):
%   host        eps_r = 9, sigma = 1e-4 S/m (above and below the layer)
%   layer       eps_r = 80, sigma = 0.5 S/m, thickness d = 0.02 m
%   antennas    both 5.99 m from the layer's front face: the FDTD model puts the source at
%               x = 0.5 m and the layer's center at x = 6.5 m, so the front face of a 0.02 m
%               layer is at 6.49 m. zsrc = zrec = 6 - 0.01*1 writes this as 6 m minus half the
%               aperture in 0.01 m cells.
%   offset      r = 1.0 m, inline (phi = 0), as in the FDTD model (source (0.5, 4.0, 4.0),
%               receiver (0.5, 4.0, 3.0) in the 8 x 8 x 8 m model)
%   source      100 MHz Ricker, frequencies (1:256)*600 MHz/256, 8x oversampled in time
% Only the J0 part of the Sommerfeld integral is evaluated. The J2 part is left out because it
% does not contribute for this geometry; j2_term_check.py evaluates its size.
%
% Output: vertical_plannar_2cm_analy_trace.csv, column 1 time (ns), column 2 field (V/m),
% read by plot_traces.py.
%
% Requires hfhorxrall.m on the MATLAB path. Runs as a script (it clears the workspace).

clear variables
clc
zsrc=6.-0.01*1;
zrec=6.-0.01*1;
z=zrec+zsrc;
hmin=zsrc-zrec;
r=1.0;
% inline antennas
phi=0;
% one offset only
nr=length(r);
nf=256;
% electric permittivity
epsr0=9;
epsr1=80;
epsr2=9;
% conductivity
sigma=0.5;
% fracture thickness
d=2e-2;
% free space parameters
c0=299792458;
mu=4e-7*pi;
eps0=1/(mu*c0^2);
% centre frequency
fc=1.e8;
% frequency values
freq=(1:nf)*6*fc/nf;
% oversampling factor in time
mf=8;
% time step
dt=1/(mf*max(real(freq)));
% time=(0:2*nf)*dt*1e9;
time = (0:mf*nf-1) * dt * 1e9;
wav=2*(freq./fc).^2.*exp(-(freq./fc).^2)/(fc*sqrt(pi));
% frequency dependent parameters eta and zeta
smu=2i*pi*freq*mu;
eta0=1e-4 + 2i*pi*freq*eps0*epsr0;
eta1=sigma + 2i*pi*freq*eps0*epsr1;
eta2=eta0;
% present arrays to zero
exx=zeros(1,nf+1);
vc1=zeros(1,nf);
% vc2=zeros(1,nf);
% set tolerance for integrations
atol=1e-16;
rtol=1e-8;
% bounds for integrals
tic
for ir=1:nr
    for jf=1:nf
        a=imag(sqrt(eta0(jf)*smu(jf)));
        b=imag(sqrt(eta1(jf)*smu(jf)));
        mxbnd=sqrt((40/abs(z))^2+abs(eta1(jf)*smu(jf)));
        vc=1; % J0-integral
        vc1(jf)=integral(@(kappa)hfhorxrall(kappa,eta0(jf),eta1(jf),eta2(jf),smu(jf),r(ir),z,d,vc)...
                    ,0,a,'reltol',rtol);
        % for wave numbers that create evanescent waves in the host medium, there
        % is no contribution from the evanescent waves at large propagation distance
        vc1(jf)=vc1(jf) + integral(@(kappa)hfhorxrall(kappa,eta0(jf),eta1(jf),eta2(jf),smu(jf),r(ir),z,d,vc)...
                    ,a,b,'reltol',rtol);
        if mxbnd > b
        vc1(jf)=vc1(jf) + integral(@(kappa)hfhorxrall(kappa,eta0(jf),eta1(jf),eta2(jf),smu(jf),r(ir),z,d,vc)...
                    ,b,mxbnd,'reltol',rtol);
        end

        %%%%%%% phased shift for arrival time correction %%%%%%%%%
        phase_shift = 1i * imag(sqrt(smu(jf)*eta0(jf)));
        vc1(jf) = vc1(jf) * exp(-phase_shift * z);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% this next part is commented out because for large vertical offset, 
% The J2-part does not contribute, only the radial part contributes 
        % vc=2; %J2-integral
        % vc2(jf)=integral(@(kappa)hfhorxrall(kappa,eta0(jf),eta1(jf),eta2(jf),smu(jf),r(ir),z,d,vc)...
        %             ,0,a,'reltol',rtol);
        % vc2(jf)=vc2(jf) + integral(@(kappa)hfhorxrall(kappa,eta0(jf),eta1(jf),eta2(jf),smu(jf),r(ir),z,d,vc)...
        %             ,a,b,'reltol',rtol);
        % if mxbnd > b
        % vc2(jf)=vc2(jf) + integral(@(kappa)hfhorxrall(kappa,eta0(jf),eta1(jf),eta2(jf),smu(jf),r(ir),z,d,vc)...
        %             ,b,mxbnd,'reltol',rtol);
        % end
    end
    exx0(2:nf+1)= vc1.*wav.*exp(-2i*pi*freq/fc)./(8*pi);
    % exx2(2:nf+1)= - cos(2*phi).*vc2.*wav.*exp(-2i*pi*freq/fc)./(8*pi);
end
exx0t=2*mf*nf*real(ifft(exx0,mf*nf))*real(freq(1));
% exx2t=2*mf*nf*real(ifft(exx2,mf*nf))*real(freq(1));
% exxt=2*mf*nf*real(ifft(exx0+exx2,mf*nf))*real(freq(1));
mx1=max(max(abs(exx0t(1,:,:))));
toc
% arrival time correction, because the time needed for the large vertical
% offset is removed inside the function hfhorxrall
tarr=sqrt(z^2+r^2)/c0;
% plot the reflected field and the incident field
figure(1)
plot(time(1:mf*nf),exx0t(1:mf*nf),'b','linewidth',2)
axis([0 200 -mx1 mx1])
xlabel('time (ns)')
ylabel('electric field amplitude (V/m)')
title(['reflected field, offset is 1.0 m, \epsilon_r =',num2str(epsr1,2),' \phi = ',num2str(phi*180/pi,2),'^o'])
set(gca,'fontsize',18)
% compute incident field (whole space solution)
Rm=sqrt(r(1)^2+hmin^2);
grg0=sqrt(smu.*eta0);
grg1=sqrt(smu.*eta1);
exxws(2:nf+1)=( ((3*(r/Rm).^2-1).*(1+grg0.*Rm)./Rm.^3 + ...
    ((r/Rm).^2-1).*grg0.^2/Rm ).*exp(-grg0*Rm) ).* ...
    wav.*exp(-2i*pi*freq/fc)./(4*pi*eta0);
exxwst=2*real(ifft(exxws,mf*nf,2))/dt;
mxw=max(abs(exxwst(1:nf)));
figure(2)
    plot(time(1:nf),exxwst(1:nf),'b','linewidth',2)
    axis([0 35 -mxw mxw])
    xlabel('time (ns)')
    ylabel('electric field amplitude (V/m)')
    title(['incident field, offset is 1.0 m, \epsilon_r =',num2str(epsr1,2),' \phi = ',num2str(phi*180/pi,2),'^o'])
    set(gca,'fontsize',18)

% =========================================================
%       SAVE DATA FOR PYTHON/GPRMAX COMPARISON
% =========================================================
filename_base = 'vertical_plannar_2cm_analy_trace';

% Column 1 = time (ns), column 2 = amplitude (V/m)
data_to_export = [time(:), exx0t(:)]; 

% Use writematrix (newer MATLAB) or csvwrite (older MATLAB)
if exist('writematrix', 'file')
    writematrix(data_to_export, [filename_base '.csv']);
else
    csvwrite([filename_base '.csv'], data_to_export);
end

fprintf('\nData export successful!\n');
fprintf('Saved to: %s.csv\n', filename_base);
% =========================================================