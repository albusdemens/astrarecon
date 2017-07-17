dir = '/data/visitor/ma2285/id11/sam8/sam8_g11_/sam8_g11_tt6_'
name = 'sam8_g11_tt6_'
num_omega = 90
num_theta = 21
bb = [669 200 600 400]
sum = true

if sum == false

    %%dark%%
    for ii=1:90
        %ii
     d_stack(:,:,ii) = edf_read(sprintf('%s/%s%04d.edf',dir,name,(ii-1)*num_theta),bb);
    end
    dark = median(d_stack,3);



    stack = zeros(bb(4),bb(3),num_omega);
    for ii=1:num_omega
        ii
        im = zeros(bb(4),bb(3));
        for jj=1:num_theta
            n = (ii-1)*num_theta+jj-1%;
            im = im + edf_read(sprintf('%s/%s%04d.edf',dir,name,n),bb)-dark;
        end
        %im = medfilt2(im);
         imshow(im,[0 max(max(im))])
         drawnow
        stack(:,:,ii) = im;
    end
end
stack(stack<0)=0;
grain.stack=stack;


nproj=90;  % number of images - image numbers start with 0 and run until nproj-1, typically
range = 360;
pixelsize=0.0007;
omega = [0 : 4 : 358];
eta=0;
theta=4.33;
dist=7.5;
offset = 0;

for n=1:size(omega,2)
        Omega = omega(n);

        rotdir0=[-sind(theta), 0, cosd(theta)];

        beamdir0 = [cosd(2*theta) 0 sind(2*theta)];

        detpos0=[dist,offset, dist*tand(2*theta)];
        detdiru0 = [0 1 0];
        detdirv0 = [0 0 -1];

        rotcomp = gtMathsRotationMatrixComp(rotdir0, 'row');



        % rotate these cordinates
        Rottens = gtMathsRotationTensor(-Omega, rotcomp);   % minus Omega because the detector and beam move - not 

        detdiru = detdiru0 * Rottens;
        detdirv = detdirv0 * Rottens;
        detpos  = detpos0 * Rottens;
        r       = beamdir0 * Rottens;
        V(n,:)=[r(1), r(2), r(3), detpos(1), detpos(2), detpos(3), detdiru(1), detdiru(2), detdiru(3), detdirv(1), detdirv(2), detdirv(3)];
end



%% save the projection stack and geometry information into the grain.mat file...

grain.geom=V;

%save('tt.mat','grain');


%end
%excluded=[ 3 5 10 21 23 29 31 32 33 37 38 42 45 47 50 53 54 60 66 74 76 77 78 82 87 90]
% grain.stack(:,excluded,:)=[];
% grain.geom(excluded,:)=[];
grain.num_rows   =size(stack,3);
grain.num_cols   =size(stack,1);
grain.vol_size_x = grain.num_rows;
grain.vol_size_y = grain.num_rows;
grain.vol_size_z = grain.num_cols;
grain.num_iter   = 50;

% 
% for ii = 1:size(grain.stack,2)
%     im = squeeze(grain.stack(:,ii,:));
%     imshow(im,[]);
%     drawnow;
%     M(n)=getframe;
% end

display('reconstructing')
volume = gtAstra3D([],[],[],grain);