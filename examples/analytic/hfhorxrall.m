function exyz=hfhorxrall(kappa,eta0,eta1,eta2,smu,r,a,d,vc)
% HFHORXRALL  Sommerfeld-integral kernel for two horizontal electric dipoles over a thin layer.
%   exyz = hfhorxrall(kappa, eta0, eta1, eta2, smu, r, a, d, vc) returns the integrand, at
%   radial wavenumbers kappa, of the reflected field of a horizontal electric dipole recorded
%   by a parallel dipole, for a layer of thickness d between two half-spaces.
%     eta0, eta1, eta2  sigma + i*omega*eps of the host above, the layer, and the host below
%     smu               i*omega*mu
%     r                 horizontal offset (m);  a  total vertical path zsrc + zrec (m)
%     vc                1 for the J0 integral, 2 for the J2 integral
%   The TE and TM generalized reflection coefficients glre, glrm are the three-layer
%   (thin-layer) coefficients. The factor exp(-arg*a) removes the host's propagation phase
%   along the vertical path, which exxgprt.m restores after the integration, so that the
%   integrand does not oscillate over the large vertical distance.
grg0=sqrt(smu*eta0+kappa.^2);
grg1=sqrt(smu*eta1+kappa.^2);
grg2=sqrt(smu*eta2+kappa.^2);
r0te=(grg0-grg1)./(grg0+grg1);
r1te=(grg1-grg2)./(grg1+grg2);
r0tm=(eta1*grg0-eta0*grg1)./(eta1*grg0+eta0*grg1);
r1tm=(eta2*grg1-eta1*grg2)./(eta2*grg1+eta1*grg2);
glre=(r0te+r1te.*exp(-2*grg1*d))./(1+r0te.*r1te.*exp(-2*grg1*d));
glrm=(r0tm+r1tm.*exp(-2*grg1*d))./(1+r0tm.*r1tm.*exp(-2*grg1*d));
J0kc=besselj(0,kappa*r).*kappa;
J2kc=besselj(2,kappa*r).*kappa;
arg=grg0-1i*imag(sqrt(smu*eta0));
if vc == 2  % J2-integral
        exyz=J2kc.*exp(-arg*a).*(grg0.*glrm/eta0+smu*glre./grg0);
else % J0-integral
        exyz=J0kc.*exp(-arg*a).*(grg0.*glrm/eta0-smu*glre./grg0);
end
