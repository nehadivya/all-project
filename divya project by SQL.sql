create database divya;
use divya;
select * from miniproject;

###null values####
SELECT * FROM MINIPROJECT WHERE 'datefobill' is NULL OR 'QUANTITY' IS NULL;

# deleting all rows that contain nulls in any of the columns
delete from miniproject
where 'typeofsales' is null
or 'patient_id' is null
or 'specialisation' is null
or 'dept' is null
or 'dateofbill' is null
or 'quantity' is null
or 'returnquantity' is null
or 'final_cost' is null
or 'final_sales' is null
or 'rtnmrp'  is null
or 'formulation'  is null
or 'drugname' is null
or 'subcat' is null
or 'subcat1' is null;

##########
select * FROM MINIPROJECT 
WHERE TYPEOFSALES='RETURN';

select  distinct 'SEVOFLURANE' from miniproject;
select distinct drugname from miniproject where drugname like 'SEVOFLURANE';
##EXTRACT THE YEAR FROM DATEofbill COLUMN#
select year ('dateofbill') from miniproject order by year ('dateofbill');
select* from miniproject;

select count(*) from miniproject;
###########duplicates value#####
select distinct patient_id from miniproject;
##########
select dateofbill,final_sales from miniproject where dateofbill between '7/15/2022' and '2/10/22';
#######sum()####
select sum(patient_id) from miniproject;
###max###
select max(quantity) from miniproject;

#mean    
SELECT AVG(final_cost) AS mean_final_cost
FROM miniproject;
###### mode #####
select quantity AS mode_quatity , count(*)
FROM miniproject group by quantity order by count(*) desc 
limit 3;


#average () FUNCTION
select * from miniproject;
select avg (final_sales) as avgsale from miniproject;
select avg (final_cost) as avgcost from miniproject;
select avg (patient_id) as totalpatient from miniproject;

# Second Moment Business Decision/Measures of Dispersion
# Variance
SELECT VARIANCE(final_cost) AS final_cost_variance
FROM miniproject;

# Standard Deviation 
SELECT STDDEV(final_sales) AS final_sales_stddev
FROM miniproject;

# Range
SELECT MAX(quantity) - MIN(quantity) AS final_cost_range
FROM miniproject;

 #Third and Fourth Moment Business Decision
-- skewness and kurkosis 
SELECT
    (
        SUM(POWER(final_sales - (SELECT AVG(final_sales) FROM miniproject), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(final_sales) FROM miniproject), 3))
    ) AS skewness,
    (
        (SUM(POWER(final_sales - (SELECT AVG(final_sales) FROM miniproject), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(final_sales) FROM miniproject), 4))) - 3
    ) AS kurtosis
FROM miniproject;

