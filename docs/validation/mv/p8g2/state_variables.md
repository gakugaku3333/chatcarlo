# P8g-2 の状態と寿命

正典: `p8g_kernel.h` の `p8g_history`（50–260 行）。§10 を優先する。
故障 g1 は `annihilated` を E で false に戻す。未初期化変数を使わない。

## 遷移をまたいで保持する値

| 値（同名は元の局所変数） | 保持の理由・寿命 |
|---|---|
| `stack`, `state`, `top` | 粒子スタックと現在の粒子。P の取り出しから C/E/H まで。失敗時も破棄しない |
| `steps`, `failed` | P/C の上限判定と H の unresolved の確定 |
| `touch[4]` | P/C の付与範囲を H の加算に使う |
| `M` | 初期材料の電子質量。E/H まで保持 |
| `hn`, `expected_n`, `de`, `radiation` | 制動放射の履歴内の順次加算。H のモーメントに使う |
| `source_energy`, `dep`, `escaped`, `rest`, `positron_escape_rest` | 履歴の収支。失敗位置の更新まで保持 |
| `cached_ev`, `arate`, `S`, `rate`, `total`, `G1`, `brate`, `bsoft` | 荷電粒子取り出しで初期化し C の周回を越えて保持。境界では cached_ev=-1 |
| `annihilated`, `suspended` | 荷電粒子取り出しで false。C の後の E で消滅の要否を判定 |
| `rng`（元の `s` の実体） | Philox の状態と calls。履歴番号ごとに初期化 |
| `local`（`P8gLocal`） | bins/coarse/fine/additions、source/dep/calls/nspec_index、events/transitions/event_used/skipped を所有 |
| `record`（呼び出し元の `P8gRecord`） | 遷移間の出力の累積、H の確定後 D で駆動側が書き出す |
| `maxdepth`（元の `P8gOutput.maxdepth`） | P/C/E の最大スタック深さを H へ渡す |
| `diag_used`, `diag_overflow`（元の `P5Diagnostic.used/overflow`） | P/C の診断の追記位置。overflow は long long のまま保持 |
| `phase` | P/C/E/H/D の駆動状態 |
| `resume_top`, `resume_kind`, `boundary_pending` | 検査の到達証拠用。親の再取り出しと境界後の再計算を数える。輸送には使わない |

## 各呼び出しの中で作り直すもの（持ち越さない）

| 範囲 | 全局所変数・別名 | 理由 |
|---|---|---|
| 初期化 | `u1,u2,j,width,cell` | スペクトル抽選は初期化内で完結 |
| 共有の別名 | `hs,fine,bins,a,counts` | 所有者は local/state/output。同一呼び出し内で結び直す |
| 材料 | `m,t` | P/C 内で材料を作り直すとき t=&m.p も再設定。t を状態に置かない |
| P（光子） | `E,c,mu,bb,nr,flight_u,tau,flight,length,escaped_photon,region,select,r,process,p,T,element,shell,binding,local_energy,emitted[32],trace[24],events,used,k,j,z,first,second,u` | 光子の飛行、緩和、二次粒子の push は P の中で完結。内部ループを分割しない |
| C（1 ステップ） | `region,ev,q[4],hardrate,hardstop,Sq,rateq,G1q,brateq,bsoftq,arateq,eb,mb,hb,bb,nr,rb,step,ab,z0,loss,event,tr,v,info,cell,prev,u,mu,angle,knew,W,dp,ds,k,mp,ms,phi,T,excess,photon_k,theta,g[8],z` | ステップ開始時の値から毎回作る。元の q[4] は状態ポインタと区別して quantities[4] と改名 |
| E | `v[3]` | 停止時消滅の方向生成から push まで E 内で完結 |
| H | `unresolved,z,p8_error,residual,coarse_error,fine_error,dn` | 最終集計内で完結。unresolved は現在の粒子、stack[0] から top の順 |
| 呼び出しの引数 | `data,meta,constants,g,s,batch,maxsteps,o,diagnostic,local,injected` | データ・形状は駆動側が全履歴を越えて所有。状態には短命なポインタを保存しない |
| 出力のビュー | `P8gOutput`, `P5Diagnostic`, 記録位置の `b,c,f` | その呼び出しの record から結び直す。共有出力の CPU 経路でも同じ遷移を使う |

元の `for` の添字（j/k/z）および上の同名の各ブロック内宣言はいずれも当該段階内で完結する。

## 補充時のゼロ化

駆動側は D を検出して、書き出しを 1 回行い、次の履歴番号を取得して begin を呼ぶ。
begin は bins/additions の nb、coarse の nc、fine の nf、record の実使用幅をゼロ化する。
診断が読む 14 要素も record のゼロ化に含む。touch、収支・モーメント、top/steps/failed、
source/nspec_index/event_used/skipped、diag_used/overflow/maxdepth と検査用の標識を初期化する。
cached_ev と係数群、annihilated/suspended は荷電粒子を取り出す際に初期化する。
スタックは初回も補充時も全体をゼロ化せず push ごとに書く。
スロットの初回初期化では phase と bins の実使用範囲だけを書き、残りは begin で初期化する。
events/transitions の未使用部分は読まず、event_used の範囲だけが有効。
H は輸送の結果と記録を確定して D に移るだけであり、補充・書き出しはしない。
CPU の保存バッチ経路は遷移から共有出力へ直接足す（特に brems[1] の加算順を保持）。

## break / continue の対応（凍結した元ファイルの行番号）

| 元の行 | 元の文 | 対応 |
|---|---|---|
| 67 | `if(steps>maxsteps) {failed=true;++counts[5];break;}` | 失敗は P→H（現在の粒子とスタック保持）、continue は P→P |
| 78 | `dep+=E;continue;` | 失敗は P→H（現在の粒子とスタック保持）、continue は P→P |
| 85 | `if(bb>flight)break;` | 光子の飛行ループ内の break のまま（P 内） |
| 86 | `if(nr<0 &#124;&#124; nr>=g->nl){escaped_photon=true;break;}` | 光子の飛行ループ内の break のまま（P 内） |
| 92 | `if(escaped_photon){if(a[17]>=2)++o->fluorescence_counts[1];if(a[17]==3)++o->fluorescence_counts[5];escaped+=E;++counts[7];continue;}` | 失敗は P→H（現在の粒子とスタック保持）、continue は P→P |
| 97 | `if(top+3>=P8G_STACK_CAP) {failed=true;++counts[6];break;}` | 失敗は P→H（現在の粒子とスタック保持）、continue は P→P |
| 107 | `if(top+used+2>=P8G_STACK_CAP){failed=true;++counts[6];break;}` | 失敗は P→H（現在の粒子とスタック保持）、continue は P→P |
| 127 | `if(top+1>o->maxdepth)o->maxdepth=top+1;continue;` | 失敗は P→H（現在の粒子とスタック保持）、continue は P→P |
| 133 | `if(steps>maxsteps) {failed=true;++counts[5];break;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 135 | `if(region<0 &#124;&#124; region>=g->nl) {escaped+=a[6];if(a[19]==1) {++counts[13];positron_escape_rest+=2*M;}break;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 138 | `dep+=a[6];a[6]=0.;break;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 155 | `if(!isfinite(step) &#124;&#124; step<0) {failed=true;++counts[6];break;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 174 | `}else a[16]=nr;continue;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 178 | `dep+=a[6];a[6]=0.;break;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 181 | `if(a[7]<=.01+1e-14) {a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;continue;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 183 | `a[9]=a[11]-a[10];a[8]=prev+a[10];continue;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 190 | `++counts[1];continue;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 193 | `p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01)continue;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 199 | `if(top+2>=P8G_STACK_CAP) {a[6]+=W;failed=true;++counts[6];break;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 201 | `if(top+1>o->maxdepth)o->maxdepth=top+1;break;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 204 | `p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01)continue;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 208 | `if(excess &#124;&#124; photon_k>T) {failed=true;++counts[6];break;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 213 | `if(top+2>=P8G_STACK_CAP) {failed=true;++counts[6];break;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 216 | `if(top+1>o->maxdepth)o->maxdepth=top+1;suspended=true;break;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 219 | `double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);continue;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 220 | `if(top+3>=P8G_STACK_CAP) {failed=true;++counts[6];break;}` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 225 | `if(top+1>o->maxdepth)o->maxdepth=top+1;break;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 228 | `dep+=a[6];a[6]=0.;break;` | 失敗は C→H（E を飛ばす）、正常 break は C→E、continue は C→C |
| 230 | `if(failed)break;` | C 直後の failed は C→H に吸収。E の容量失敗は E→H |
| 232 | `if(top+3>=P8G_STACK_CAP) {a[6]=0.;failed=true;++counts[6];break;}` | C 直後の failed は C→H に吸収。E の容量失敗は E→H |

H→D、空のスタックの P→H、通常の E→P は明示的な遷移とする。
硬衝突の容量失敗の a[6]+=W、制動放射の容量失敗前の brems への加算は元の順に残す。
光電効果の緩和、top+used+2 判定、付与、光子 push、光電子 push は P 内で一まとまりに保つ。

CPU の模擬と GPU は周回冒頭の phase を snapshot し、全段階のマスクを仕事前に確定する。
初回・補充の取得は next=0 から。補充した履歴は次の周回から進む。
検査ビルドの hits は CPU の呼び出し元の配列、GPU の cp.zeros のグローバルメモリ。
GPU の局所配列に atomicAdd しない。輸送の指紋には遷移・駆動・出力を含め、
Colab/CLI の起動・測定のスクリプトは含めない。
