% 
% obs3_x = expert_sc(:,3)*5;
% obs3_y = expert_sc(:,4)*5;
% 
% obs3_x1 = expert_sc(:,21)*5;
% obs3_y1 = expert_sc(:,22)*5;
% 
% obs3_x2 = expert_sc(:,39)*5;
% obs3_y2 = expert_sc(:,40)*5;

obs4_x = expert_sc(:,3)*5;
obs4_y = expert_sc(:,4)*5;

obs4_x1 = expert_sc(:,27)*5;
obs4_y1 = expert_sc(:,28)*5;

obs4_x2 = expert_sc(:,51)*5;
obs4_y2 = expert_sc(:,52)*5;
% 
% obs6_x = expert_sc(:,3)*5;
% obs6_y = expert_sc(:,4)*5;
% 
% obs6_x1 = expert_sc(:,39)*5;
% obs6_y1 = expert_sc(:,40)*5;
% 
% obs6_x2 = expert_sc(:,75)*5;
% obs6_y2 = expert_sc(:,76)*5;
% 
% 
histogram2(obs4_x,obs4_y,[25 25],'FaceColor','flat','Normalization', 'probability')
hold on 
histogram2(obs4_x1,obs4_y1,[25 25],'FaceColor','flat','Normalization', 'probability')
histogram2(obs4_x2,obs4_y2,[25 25],'FaceColor','flat','Normalization', 'probability')

xlabel('x [m]')
ylabel('y [m]')
set(gca, 'FontSize',15)
set(gca,'FontWeight','bold');
colorbar