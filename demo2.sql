-- 1. 查看数据概览
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT College_ID) as unique_colleges
FROM college_student_placement_dataset;

-- 2. 检查缺失值
SELECT 
    SUM(CASE WHEN IQ IS NULL THEN 1 ELSE 0 END) as missing_IQ,
    SUM(CASE WHEN CGPA IS NULL THEN 1 ELSE 0 END) as missing_CGPA,
    SUM(CASE WHEN Placement IS NULL THEN 1 ELSE 0 END) as missing_Placement
FROM college_student_placement_dataset;

-- 3. 查看录用与未录用的人数
SELECT 
    Placement,
    COUNT(*) as count
FROM college_student_placement_dataset
GROUP BY Placement;

-- 4. 按学院统计录用率
SELECT 
    College_ID,
    AVG(CASE WHEN Placement = 'Yes' THEN 1 ELSE 0 END) * 100 as placement_rate
FROM college_student_placement_dataset
GROUP BY College_ID
ORDER BY placement_rate DESC;

