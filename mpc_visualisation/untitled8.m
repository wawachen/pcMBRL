clear
for i = 1:5:35
    figure
    states = importdata(sprintf('mpc_visualisation%d.mat',i));
    costs = importdata(sprintf('mpc_costs%d.mat',i));
    [b,index] = sort(costs);
    
    plot(states(:,index(1),1)*5,states(:,index(1),2)*5,'Linewidth',1.8,'Color','r')
    hold on 
    plot(states(:,index(2),1)*5,states(:,index(2),2)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(3),1)*5,states(:,index(3),2)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(4),1)*5,states(:,index(4),2)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(5),1)*5,states(:,index(5),2)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(6),1)*5,states(:,index(6),2)*5,'Linewidth',1.5,'Color','k')
    
    % plot(states(:,index(46),1)*5,states(:,index(46),2)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(47),1)*5,states(:,index(47),2)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(48),1)*5,states(:,index(48),2)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(49),1)*5,states(:,index(49),2)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(50),1)*5,states(:,index(50),2)*5,'--','Linewidth',1.5,'Color','k')
    
    viscircles([states(1,1,1)*5,states(1,1,2)*5],0.1,'Color','r')
    hold on 
    
    
    
    plot(states(:,index(1),3)*5,states(:,index(1),4)*5,'Linewidth',1.8,'Color','g')
    
    hold on 
    plot(states(:,index(2),3)*5,states(:,index(2),4)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(3),3)*5,states(:,index(3),4)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(4),3)*5,states(:,index(4),4)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(5),3)*5,states(:,index(5),4)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(6),3)*5,states(:,index(6),4)*5,'Linewidth',1.5,'Color','k')
    
    % plot(states(:,index(46),2)*5,states(:,index(46),3)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(47),2)*5,states(:,index(47),3)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(48),2)*5,states(:,index(48),3)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(49),2)*5,states(:,index(49),3)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(50),2)*5,states(:,index(50),3)*5,'--','Linewidth',1.5,'Color','k')
    
    viscircles([states(1,1,3)*5,states(1,1,4)*5],0.1,'Color','g')
    
    
    
    
    plot(states(:,index(1),5)*5,states(:,index(1),6)*5,'Linewidth',1.8,'Color','b')
    
    hold on 
    plot(states(:,index(2),5)*5,states(:,index(2),6)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(3),5)*5,states(:,index(3),6)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(4),5)*5,states(:,index(4),6)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(5),5)*5,states(:,index(5),6)*5,'Linewidth',1.5,'Color','k')
    plot(states(:,index(6),5)*5,states(:,index(6),6)*5,'Linewidth',1.5,'Color','k')
    
    % plot(states(:,index(46),4)*5,states(:,index(46),5)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(47),4)*5,states(:,index(47),5)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(48),4)*5,states(:,index(48),5)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(49),4)*5,states(:,index(49),5)*5,'--','Linewidth',1.5,'Color','k')
    % plot(states(:,index(50),4)*5,states(:,index(50),5)*5,'--','Linewidth',1.5,'Color','k')
    
    
    viscircles([states(1,1,5)*5,states(1,1,6)*5],0.1,'Color','b')
    
    rectangle('Position',[2.8-0.5,0-0.5,1,1],'Curvature',[1,1],'EdgeColor','b','LineWidth',1.8,'LineStyle','--'),axis equal;
    rectangle('Position',[0-0.5,0-0.5,1,1],'Curvature',[1,1],'EdgeColor','r','LineWidth',1.8,'LineStyle','--'),axis equal;
    rectangle('Position',[-2.8-0.5,0-0.5,1,1],'Curvature',[1,1],'EdgeColor','g','LineWidth',1.8,'LineStyle','--'),axis equal;
    
    set(gca,'linewidth',1.8);
    set(gca, 'FontSize',15)
    set(gca,'FontWeight','bold');
    legend("UAV1",'','','','','',"UAV2",'','','','','',"UAV3",'','','','','')
    xlabel("x (m)")
    ylabel("y (m)")
    xlim([-5,5])
    ylim([-5,5])
    
%     figure
%     plot(states(:,1,1)*5,'-.o','color','r','MarkerFaceColor','r','LineWidth',1.5)
%     hold on 
%     plot(states(:,1,2)*5,'b-o')
end