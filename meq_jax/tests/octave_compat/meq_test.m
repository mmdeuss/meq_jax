% Octave-compatible shim exposing the static helpers of meq/tests/meq_test.m
% that MEQ-JAX tests rely on. The original class inherits from
% matlab.unittest.TestCase, which GNU Octave cannot load; this shim contains
% verbatim copies of the self-contained static methods
% getCircularEquilibrium and generate_flux_map.
%
% [+MEQ MatlabEQuilibrium Toolbox+]

%    Copyright 2022-2025 Swiss Plasma Center EPFL and contributors
%
%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%
%       http://www.apache.org/licenses/LICENSE-2.0
%
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.

classdef meq_test

  methods (Static)
    function [L,LY] = getCircularEquilibrium(L,R0,Z0,FA,FB,rBt)
      % Flux map and plasma domain
      Fx = ((L.rx'-R0).^2 + (L.zx-Z0).^2)*(FB-FA) + FA; % Dummy flux map - initial guess
      [qdl,kl] = min((L.G.rl-R0).^2 + (L.G.zl-Z0).^2);
      rB = L.G.rl(kl); zB = L.G.zl(kl);
      Fx = FA + 1/qdl*(Fx-FA); % Rescale to get correct FB
      a0 = sqrt(qdl);          % Plasma minor radius
      
      Fy = Fx(2:end-1,2:end-1); % on inner grid
      FyN = (Fy-FA)/(FB-FA);
      Opy = int8(FyN>=0 & FyN<=1); % logical where there is plasma
      
      % Current density
      Imax = 1e4*sign(FA-FB);
      Iy = Imax*(1-FyN).*double(Opy>0);
      
      % Pack all quantities
      LY.Ip = sum(Iy(:));
      LY.Fx = Fx;
      LY.Iy = Iy;
      LY.FA = FA;
      LY.FB = FB;
      LY.Opy = Opy;
      LY.rA = R0;
      LY.zA = Z0;
      LY.dr2FA = 2/a0.^2*(FB-FA);
      LY.dz2FA = 2/a0.^2*(FB-FA);
      LY.drzFA = 0;
      LY.rB = rB;
      LY.zB = zB;
      LY.lX = false;
      LY.rX = zeros(0,1);
      LY.zX = zeros(0,1);
      LY.rBt = rBt;
      
      % Flux surfaces positions
      %   Fx = [(rx-R0)^2 + (zx-Z0)^2]/a0^2*(FB-FA) + FA
      %   circular flux surfaces centered at (R0,Z0) with minor radius a0*rho
      %   with rho the square root of the normalized poloidal flux
      aq = a0.*L.pq.*ones(L.noq,1);
      rq = aq.*L.crq + R0; zq = aq.*L.czq + Z0;
      irq = 1./rq;
        
      % Toroidal field profile (TT'=0)
      iTQ = 1/rBt*ones(L.npq+1,1);
        
      % Flux surface integrals
      % Note for the flux surface averages:
      %   <f> = d/dV int(f dV)
      %       = d/dV int(f R dR dZ dphi)
      %   For nested tori of minor radius a0*rho
      %   <f> = 2 pi d/dV int(f R a0^2 rho drho dtheta)
      %   Since V = 2 pi R0 pi (a0 rho)^2, dV = 4 pi^2 R0 a0^2 rho drho
      %   <f> = 1/(2 pi R0) int(f R dtheta)
      FN = L.pQ.^2';
      % Q0Q = <1/R> = 1/(2 pi R0) int(dtheta) = 1/R0
      Q0Q = 1/R0 * ones(L.npq+1,1); 
      % Q1Q = -dpsi/dV
      %   V = 2 pi R0 pi (a0 rho)^2
      %   psi = FA + (FB-FA) rho^2 = FA + (FB-FA) V/(2 pi R0 pi a0^2)
      % Q1Q = -(FB-FA)/(2 pi R0 pi a0^2)
      Q1Q = -(FB-FA)/(2*pi*pi*R0*a0*a0)*ones(L.npq+1,1);
      % Q2Q = <1/R^2>
      % Q2Q = 1/(2 pi R0) int(dtheta/(R0 - a0 rho cos(theta))
      %     = 1/R0^2 1/sqrt(1-(a0 rho / R0)^2)
      Q2Q = 1./(R0^2*sqrt(1-(a0/R0)^2*FN));
      % Q3Q = <|grad psi|^2/R^2>
      % |grad psi|^2 = |dpsi/dR|^2 + |dpsi/dZ|^2 = 4 (FB-FA)^2 rho^2 / a0^2
      % So |grad psi| only depends on rho and not on theta
      % Q3Q = |grad psi|^2 <1/R^2>
      % Q3Q = 4 (FB-FA)^2 rho^2 / a0^2 Q2Q
      Q3Q = (2*(FB-FA)*L.pQ.'/a0).^2.*Q2Q;
      % Q4Q = <|grad psi|^2>
      % Q4Q = |grad psi|^2 <1>
      % Q4Q = 4 (FB-FA)^2 rho^2 / a0^2
      Q4Q = (2*(FB-FA)*L.pQ.'/a0).^2;
      % iqQ = 1/q = -2 pi/(T dV/dpsi <1/R^2>)
      iqQ = 2*pi*Q1Q./Q2Q.*iTQ;
      % ItQ = 1/mu0 int(Bp dlp) = 1/mu0
      ItQ = Q3Q./Q1Q./mu0./4/pi^2;
      % LpQ = int(dlp)
      %   poloidal circumference of a torus
      LpQ = 2*pi*a0*L.pQ.';
      % rbQ = int(rdlp)/int(dlp)
      %   poloidal barycenter of a torus
      rbQ = R0 * ones(L.npq+1,1);
      % Q5Q = <|grad psi|/2/pi>
      % Q5Q = |grad psi|/2/pi <1>
      % Q5Q = 1/pi/a0 |FB-FA| rho
      % Alternatively Q5Q = rbQ LpQ |Q1Q|
      Q5Q = abs(FB-FA)*L.pQ.'/(pi*a0);
      % SlQ - lateral area of a torus
      SlQ = (4*pi^2)*R0*a0*L.pQ.';
      % VQ  - volume of a torus
      VQ  = 2*pi^2*R0*(a0*L.pQ.').^2;
      % AQ  - cross section area of a torus
      AQ  = pi*(a0*L.pQ.').^2;
      
      LY.aq = aq;
      LY.rq = rq;
      LY.zq = zq;
      LY.irq = irq;
      LY.Q0Q = Q0Q;
      LY.Q1Q = Q1Q;
      LY.Q2Q = Q2Q;
      LY.Q3Q = Q3Q;
      LY.Q4Q = Q4Q;
      LY.iqQ = iqQ;
      LY.iTQ = iTQ;
      LY.ItQ = ItQ;
      LY.LpQ = LpQ;
      LY.rbQ = rbQ;
      LY.Q5Q = Q5Q;
      LY.SlQ = SlQ;
      LY.VQ  = VQ;
      LY.AQ  = AQ;
      
      % Gaps
      co = cos(L.G.oW)/a0;
      so = sin(L.G.oW)/a0;
      dr = (L.G.rW - R0)/a0;
      dz = (L.G.zW - Z0)/a0;
      
      % Looking for solution of a*x^2 + 2*b*x + c = 0
      a = co.^2  + so.^2;
      b = dz.*so - dr.*co;
      c = dr.^2  + dz.^2 - FB;
      aW1 = (-b+sqrt(b.^2-a.*c))./a;aW1(aW1 < 0) = Inf;
      aW2 = (-b-sqrt(b.^2-a.*c))./a;aW2(aW2 < 0) = Inf;
      aW = min(aW1,aW2);
      
      LY.aW = aW;
    end

    function [S] = generate_flux_map(type,sIp)
      % defaults
      % [ro,zo]: center coordinates
      % wo: width
      % so: +1 for gaussian, -1 for "hyperbolic gaussian" (X-point)
      % Fo: amplitude
      ro = 1; zo = 0.4; wo = 0.2; so = +1; % gaussian flux distribution parameters
      Fo = 0.01; % up/down flux level
      switch type
        case 'Vacuum'
          so = -1;
          ndom = 0; nA = 0; nX = 1; lX = false(0,1);
        case 'SND'
          Fo = Fo*[1,1]; ro = [ro,0.6]; zo = [0,-0.8];
          wo = wo*[1,0.4];
          ndom = 1; nA = 1; nX = 1; lX = true;
        case 'DND'
          Fo = Fo*[1,1,1]; ro = [ro,0.6,0.6]; zo = [0,-0.8,0.8];
          wo = wo*[1,0.4,0.4];
          ndom = 1; nA = 1; nX = 2; lX = true;
        case 'Lim'
          zo = 0;
          ndom = 1; nA = 1; nX = 0;  lX = false;
        case 'Lim-X'           % limited with a non-limiting X-point
          Fo = Fo*[1,1]; ro = [ro,0.6]; zo = [zo-0.2,-0.8];
          ndom = 1; nA = 1; nX = 1; lX = true;
        case 'Double-Snowflake-Minus'
          ndom = 1; nA = 1; nX = 4; lX = true;
          Fo = Fo*[1.6,1,1,1,1];
          ro = [ro,0.6,0.6,1.4,1.4];
          zo = [0,-0.8,0.8,-0.8,0.8];
        case 'Double-Snowflake-Plus'
          ndom = 1; nA = 1; nX = 4; lX = true;
          Fo = Fo*[1.1,1,1,1,1];
          ro = [ro,0.6,0.6,1.4,1.4];
          zo = [0,-0.8,0.8,-0.8,0.8];
        case 'Doublet'
          ro = ro*[1,1]; zo = zo*[-1,1]; wo = wo*[1,1];  Fo = Fo*[1,1];
          ndom = 3; nA = 2; nX = 1;  lX = [true;true;false];
        case 'Droplet'
          zo = (zo+0.2)*[-1,1];
          ndom = 2; nA = 2; nX = 1; lX = [false;false];
        case 'Doublet-div-nomantle'
          ro = ro*[1,1,1]; zo = [-zo,zo+0.1,-1.1]; wo = wo*[1,1,.3];  Fo = Fo*[1,1.1,2];
          ndom = 2; nA = 2; nX = 2; lX = [true;true];
        case 'Doublet-div'
          ro = ro*[1,1,1]; zo = [-zo,zo,-1]; wo = wo*[1,1,.2];  Fo = Fo*[1,1,1];
          ndom = 3; nA = 2; nX = 2; lX = [true;true;true];
        case 'Triplet'
          ro = ro*[1,1,1]; zo = 1.2*[-zo,0,zo]; wo = 0.2*wo*[1,1,1];  Fo = Fo*[2,1,1];
          ndom = 5; nA = 3; nX = 2; lX = [true;true;true;true;false];
        case 'Triplet-madness'
          ro = ro*[1,1,1,1]; zo = [1.2*[-zo-0.1,0,zo],-1]; wo = 0.2*wo*[1,1,1,1];  Fo = Fo*[1,1,1.3,1];
          ndom = 4; nA = 3; nX = 3; lX = [true;true;true;true];
        case 'Boundary-X'
          ro = [ro,0.55*2-ro+0.01]; zo = zo*[-1,1]; wo = wo*[1,1]; Fo = Fo*[1,1];
          ndom = 1; nA = 1; nX = 1; lX = true;
        otherwise
          error('unknown type %s',type)
      end
      so = repmat(so,1,numel(Fo));
      
      S.Fxh    = @(rrx,zzx) reshape(sum(sIp*(Fo                                         .*exp( -((rrx(:)-ro).^2 + so.*(zzx(:)-zo).^2)./wo)),2),size(rrx));
      S.drFxh  = @(rrx,zzx) reshape(sum(sIp*(Fo.*(-2*(rrx(:)-ro)./wo)                   .*exp( -((rrx(:)-ro).^2 + so.*(zzx(:)-zo).^2)./wo)),2),size(rrx));
      S.dzFxh  = @(rrx,zzx) reshape(sum(sIp*(Fo.*(-2*(zzx(:)-zo)./wo).*so               .*exp( -((rrx(:)-ro).^2 + so.*(zzx(:)-zo).^2)./wo)),2),size(rrx));
      S.dr2Fxh = @(rrx,zzx) reshape(sum(sIp*(Fo.*(4*(rrx(:)-ro).^2./wo.^2 -2./wo)       .*exp( -((rrx(:)-ro).^2 + so.*(zzx(:)-zo).^2)./wo)),2),size(rrx));
      S.dz2Fxh = @(rrx,zzx) reshape(sum(sIp*(Fo.*(4*(zzx(:)-zo).^2./wo.^2 -2./wo)       .*exp( -((rrx(:)-ro).^2 + so.*(zzx(:)-zo).^2)./wo)),2),size(rrx));
      S.drzFxh = @(rrx,zzx) reshape(sum(sIp*(Fo.*(4*(rrx(:)-ro).*(zzx(:)-zo)./wo.^2).*so.*exp( -((rrx(:)-ro).^2 + so.*(zzx(:)-zo).^2)./wo)),2),size(rrx));
      S.ndom = ndom;
      S.nA = nA;
      S.nX = nX;
      S.lX = lX;
    end
  end

end
