#!/bin/bash

if [ -z "$thirdparty_manage_path" ]; then
    echo "thirdparty_manage_path no set"
    exit -1
fi

Path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -rf $Path/proto_msg
mkdir -p $Path/proto_msg
msg2proto $Path/msg  $Path/proto_msg

find $Path/proto_msg/ -name "*.proto" -exec sed -i.bak 's|import "/|import "|g' {} +

rm -rf $Path/proto_msg/*.bak

mkdir -p $Path/proto_msg/cpp

# 已有导出：手臂/关节相关接口
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/joint_state.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/arm_state.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/joint_command.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/arm_command.proto

# 新增导出：末端夹爪 + 腰颈接口，以及其依赖的多维数组辅助消息
# 目的：覆盖本次要求模块（末端夹爪、腰颈）的 proto 打包
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/array1_d.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/array2_d.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/array3_d.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/gripper_state.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/gripper_command.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/head_waist_state.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/head_waist_command.proto

# 已有导出：移动平台相关接口
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/motion_platform_state.proto
$thirdparty_manage_path/bin/protoc --cpp_out=$Path/proto_msg/cpp --proto_path=$Path/proto_msg/ $Path/proto_msg/motion_platform_command.proto


#$thirdparty_manage_path/bin/protoc --dart_out=$Path/proto_msg/dart --proto_path=$Path/proto_msg/ $Path/proto_msg/*.proto
#$thirdparty_manage_path/bin/protoc --python_out=$Path/proto_msg/python --proto_path=$Path/proto_msg/ $Path/proto_msg/*.proto
