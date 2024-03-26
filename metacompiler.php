<?php
if (count($argv) < 2)
	die("Usage: {@$argv[0]} code.cu\n");

$code = file_get_contents($argv[1]);

$jobs = preg_match_all('/class\s+(\w+?)\: public AbstractJob\s*{(.*?)};/sm', $code, $matches, PREG_SET_ORDER);
// print_r($matches);
$count = count($matches);

/// Part1: Job types enum
$enum = [];
for ($i=0;$i< $count; ++$i) {
	$jobname = $matches[$i][1];
	$jobname = str_replace("Job", "", $jobname);
	$enum[] = $jobname;
}
$enum_str = "enum JobTypes {\n\tAbstract,\n\t";
$enum_str .= join(",\n\t", $enum);
$enum_str .= "\n};\n";

echo $enum_str;



/// Part2: Iter call function
$iter =[];
for ($i=0;$i<$count; ++$i)
	$iter[]= "if (job->job_type == JobTypes::{$enum[$i]})\n\t(({$enum[$i]}Job *)job)->iter(id, d_offsets, d_edges, n, m);";

$iter_str = join("\nelse ", $iter);
$iter_str .= PHP_EOL. 'else printf("Unsupported job type!\n");' . PHP_EOL;
echo $iter_str;

/// Part3: FusionJob union
$typedata = [];
for ($i=0;$i<$count;++$i) {
	preg_match('#//PRAGMA: job data\n(.*?)\n//PRAGMA#ms', $matches[$i][2], $data);
	$data=$data[1];
	$data=explode("\n", $data);
	$data=array_map('trim', $data);

	$typedata[$i] = [];
	foreach ($data as $datum) {
		preg_match('/(.*?)\s+(.*?);/', $datum, $match);
		$type = $match[1];
		$name = $match[2];
		while ($name[0] == '*')
		{
			$name = substr($name, 1);
			$type .= "*";
			if (!(substr($type, -1) == '*' or $type == 'i64' or $type =='double' or $type == 'u64'))
				echo "Warning: Type '{$type}' may not be 8 bytes. I suggest using only 8 byte types in jobs.\n";
		}
		$typedata[$i][] = ['type' => $type, 'name' => $name];
	}
}
$max = 0;
for ($i=0; $i<count($typedata); ++$i) {
	$max=max($max, count($typedata[$i]));
}
// print_r($typedata);
$union = [];
for ($i=0;$i<$max;++$i) {
	$values = [];
	for ($j=0;$j<$count; ++$j) {
		if (isset($typedata[$j][$i]))
			$values[] = $typedata[$j][$i]['type'] . " " . $typedata[$j][$i]['name'];
	}
	while (count($values) > 1 && $values[0] == $values[1]) {
		array_shift($values);
	}
	$union[$i] = $values;

}
// print_r($union);

$union_str = "struct FusionJob: public AbstractJob {\n";
foreach ($union as $u) {
	if (count($u) > 1) {
		$union_str.= "\tunion {\n";
		foreach ($u as $item) {
			$union_str .= "\t\t{$item};\n";
		}
		$union_str .= "\t};\n";
	}
	else
		$union_str.= "\t{$u[0]};\n";
}
$union_str .= "};\n";
echo $union_str;