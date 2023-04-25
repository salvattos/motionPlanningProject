%import AFPObstacle.*
%Dummy straight line path
xPath = [];
yPath = [];
xPath = linspace(0,100,100);
yPath = linspace(0,100,100).';

%ρ0 is the affected distance of obstacle, ρg(q) is the Euclidean distance 
%between the robot location and the target, ρ(q) is the minimum distance 
%between the obstacle affected area to the robot location, 
obstaclePos = [50;50];
goalPos  = [95;85];

startPos = [10;15];
xi = .100;
VG = [0;0];
V0 = [0;0];
V = [0;0];
p0 = 5;
KV = 60;

OB1 = APFObstacle(obstaclePos,goalPos,p0,V0);
Fsum = [];
for x = 1:size(xPath,2)
    for y = 1:size(yPath,1)
        currentPos = [x;y];
        %Attraction force
        Fatt = calcFatt(xi,calcpG(goalPos,currentPos),KV,VG,V);
        %Frep = OB1.calcFrepTotal(currentPos,V);
        Frep = 0;
        Fsum = Frep+Fatt;
       %pGLog(x,y) = calcpG(goalPos,currentPos);
        dxLog(x,y) = Fsum(1);
        dyLog(x,y) = Fsum(2);
    end
end

xPath = repmat(xPath,size(xPath,2),1);
yPath = repmat(yPath,1,size(yPath,1));

test = quiver(xPath,yPath,dxLog,dyLog)
hold on
plot(goalPos(1),goalPos(2),'rx')
plot(startPos(1),startPos(2),'gx')
hold off



