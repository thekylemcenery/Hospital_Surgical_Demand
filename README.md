# Hospital_Surgical_Demand

**Description**  
This project uses **K-means clustering** to segment hospital surgical cases into distinct demand groups based on features such as length of stay, theatre time, staff requirements, and consumables cost. The goal is to provide actionable insights for NHS resource planning and operational efficiency.

**Features**  
- Synthetic dataset simulating surgical admissions  
- Clustering and profiling of cases into resource-based groups  
- Calculation of cluster-level summaries, including proportion of total theatre time, length of stay, and consumables cost  
- Export of cluster summaries to Excel for further analysis and charting  

**Usage**  
1. Run the Python script to generate the synthetic dataset and perform clustering.  
2. Cluster summaries will be exported to `cluster_summary.xlsx`.  
3. Use Excel or other tools to visualize cluster profiles and create operational dashboards.  

**Insights**  
- Identify high-resource clusters that may require special scheduling or ICU allocation  
- Understand the distribution of cases and their contribution to total hospital resources  
- Support evidence-based decisions for staffing, theatre scheduling, and cost management  

**Requirements**  
- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- matplotlib (optional, for plotting)  

**License**  
MIT
