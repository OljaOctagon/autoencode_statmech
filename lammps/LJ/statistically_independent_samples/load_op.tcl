set mol top
set sel [atomselect $mol "all"]

set nf     [molinfo $mol get numframes]
set natoms [molinfo $mol get numatoms]

set fp [open "w6.dat" r]

for {set i 0} {$i < $nf} {incr i} {

    if {[gets $fp line] < 0} {
        error "q6_user.dat ended early at frame $i"
    }
    
    $sel frame $i
    $sel set user $line
}


close $fp
$sel delete

mol modcolor 0 $mol User
puts "Loaded q6 into user field. Now color by User."

