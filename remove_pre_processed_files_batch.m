clear all

list_folders = {
'L_ACA';                       
'L_Angular';                    
'L_Anterior_MCA';               
'L_Anterior_choroidal';         
'L_Basilar_perforating';        
'L_Basilar_tip';                
'L_Calcarine';                  
'L_Cerebellar';                 
'L_Inferior_MCA';               
'L_Lenticulostriate';           
'L_Long_insular_perforating';   
'L_Opercular';                  
'L_Parietal';                   
'L_Posterior_borderzone';       
'L_Posterior_choroidal';        
'L_Precentral';                 
'L_Prefrontal';                 
'L_Rolandic';                   
'L_Thalamoperforators';         
'L_Total_MCA';                  
'L_Total_PCA';                  
'R_ACA';                        
'R_Angular';                    
'R_Anterior_MCA';               
'R_Anterior_choroidal';         
'R_Basilar_perforating';        
'R_Basilar_tip';                
'R_Calcarine';                  
'R_Cerebellar';                 
'R_Inferior_MCA';               
'R_Lenticulostriate';           
'R_Long_insular_perforating';   
'R_Opercular';                  
'R_Parietal';                   
'R_Posterior_borderzone';       
'R_Posterior_choroidal';        
'R_Precentral';                 
'R_Prefrontal';                 
'R_Rolandic';                   
'R_Thalamoperforators';         
'R_Total_MCA';                  
'R_Total_PCA';                  
'outside_clusters'
};

for i=1:numel(list_folders)
    
    delete 'R_Total_PCA/r_s*';
    delete 'R_Total_PCA/_s*';
    
end